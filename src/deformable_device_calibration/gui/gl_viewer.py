# -*- coding: utf-8 -*-
# Copyright (c) 2025 Ruizhe Lin
# Licensed under the MIT License.


import ctypes

import numpy as np
from OpenGL import GL
from OpenGL.GL import shaders
from PyQt6.QtCore import QTimer, QSize, pyqtSignal, Qt
from PyQt6.QtWidgets import QSizePolicy
from PyQt6.QtOpenGLWidgets import QOpenGLWidget

VERT_SRC = """
#version 330 core
layout (location = 0) in vec2 a_pos;   // -1..1
layout (location = 1) in vec2 a_uv;    //  0..1 (v=0 bottom, v=1 top)

uniform vec2  u_scale;   // aspect-fit scaling
uniform vec2  u_center;  // texture UV at view center
uniform float u_zoom;    // >= 1.0

out vec2 v_uv;

void main() {
    gl_Position = vec4(a_pos * u_scale, 0.0, 1.0);
    v_uv = (a_uv - vec2(0.5)) / u_zoom + u_center;
}
"""

FRAG_SRC = """
#version 330 core
in vec2 v_uv;
out vec4 out_color;

uniform sampler2D u_tex;     // GL_R16 (normalized 0..1 when sampled)
uniform float u_black;       // 0..1
uniform float u_white;       // 0..1
uniform float u_gamma;       // >0

void main() {
    float v = texture(u_tex, v_uv).r;                  // 0..1 from R16
    v = (v - u_black) / max(u_white - u_black, 1e-6);  // normalize by levels
    v = clamp(v, 0.0, 1.0);
    v = pow(v, 1.0 / max(u_gamma, 1e-6));              // gamma
    out_color = vec4(v, v, v, 1.0);
}
"""


class GLGray16Viewer(QOpenGLWidget):
    """
    High-throughput uint16 grayscale viewer.
    Call .set_frame(frame_u16) from the GUI thread (or via Qt signal with QueuedConnection).
    """
    frameConsumed = pyqtSignal(object)   # token
    frameDiscarded = pyqtSignal(object)  # token when replaced before upload
    mousePixelChanged = pyqtSignal(int, int, int)  # x, y, value (uint16), -1 if invalid

    def __init__(self, use_pbo=True, parent=None):
        super().__init__(parent)
        self.use_pbo = use_pbo
        self.use_pbo = use_pbo
        self._pending = None
        self._pending_token = None
        self._display_frame = None
        self._display_token = None
        self._have_new = False

        self.setMouseTracking(True)

        self._zoom = 1.0
        self._min_zoom = 1.0
        self._max_zoom = 20.0
        self._center_u = 0.5
        self._center_v = 0.5

        self._dragging = False
        self._drag_anchor_tex_uv = None  # (tu, tv) at press

        self._sx = 1.0   # current aspect-fit scale (x)
        self._sy = 1.0   # current aspect-fit scale (y)

        # CPU-side state (avoid per-frame allocations)
        self._pending = None  # last received frame (uint16 HxW)
        self._have_new = False
        self._img_w = 0
        self._img_h = 0

        # GPU resources
        self._prog = None
        self._vao = None
        self._vbo = None
        self._tex = None

        # PBO ping-pong
        self._pbo = None
        self._pbo_idx = 0
        self._pbo_nbytes = 0

        # Display controls (in uint16 space)
        self.black_u16 = 0
        self.white_u16 = 65535
        self.gamma = 1.0

        # Drive repaint; vsync caps to monitor refresh (~60)
        self._timer = QTimer(self)
        self._timer.timeout.connect(self.update)
        self._timer.start(0)

        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

    def sizeHint(self) -> QSize:
        return QSize(700, 700)

    def minimumSizeHint(self) -> QSize:
        return QSize(300, 300)

    def _clamp_center(self):
        z = max(self._zoom, 1e-6)
        if z <= 1.000001:
            self._center_u, self._center_v = 0.5, 0.5
            return
        ext = 0.5 / z
        self._center_u = float(max(ext, min(1.0 - ext, self._center_u)))
        self._center_v = float(max(ext, min(1.0 - ext, self._center_v)))

    def _widget_pos_to_a_uv(self, px: int, py: int):
        """Map widget pixel -> quad UV (a_uv) in [0..1], v=0 bottom, v=1 top. None if in letterbox area."""
        if self._img_w <= 0 or self._img_h <= 0:
            return None
        W = max(1, self.width())
        H = max(1, self.height())

        img_aspect = self._img_w / self._img_h
        win_aspect = W / H
        if img_aspect >= win_aspect:
            sx, sy = 1.0, win_aspect / img_aspect
        else:
            sx, sy = img_aspect / win_aspect, 1.0

        ndc_x = (2.0 * px / (W - 1)) - 1.0 if W > 1 else 0.0
        ndc_y = 1.0 - (2.0 * py / (H - 1)) if H > 1 else 0.0  # y up

        if abs(ndc_x) > sx or abs(ndc_y) > sy:
            return None

        u = (ndc_x / sx + 1.0) * 0.5
        v = (ndc_y / sy + 1.0) * 0.5
        return u, v

    def _a_uv_to_tex_uv(self, u: float, v: float):
        """Apply zoom+pan mapping (must match your vertex shader)."""
        tu = (u - 0.5) / self._zoom + self._center_u
        tv = (v - 0.5) / self._zoom + self._center_v
        return tu, tv

    def _widget_pos_to_image_xy(self, px: int, py: int):
        """Widget pixel -> image indices (ix,iy), origin top-left."""
        uv = self._widget_pos_to_a_uv(px, py)
        if uv is None:
            return None
        u, v = uv
        tu, tv = self._a_uv_to_tex_uv(u, v)

        if tu < 0.0 or tu > 1.0 or tv < 0.0 or tv > 1.0:
            return None

        ix = int(tu * (self._img_w - 1) + 0.5)
        iy = int((1.0 - tv) * (self._img_h - 1) + 0.5)  # convert tex v to top-left origin
        return ix, iy

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton and self._zoom > 1.000001:
            p = event.position().toPoint()
            uv = self._widget_pos_to_a_uv(p.x(), p.y())
            if uv is not None:
                u, v = uv
                self._drag_anchor_tex_uv = self._a_uv_to_tex_uv(u, v)
                self._dragging = True
                self.setCursor(Qt.CursorShape.ClosedHandCursor)
                event.accept()
                return
        super().mousePressEvent(event)

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self._dragging = False
            self._drag_anchor_tex_uv = None
            self.unsetCursor()
            event.accept()
            return
        super().mouseReleaseEvent(event)

    def mouseMoveEvent(self, event):
        p = event.position().toPoint()

        # --- PAN while dragging ---
        if self._dragging and self._drag_anchor_tex_uv is not None:
            uv = self._widget_pos_to_a_uv(p.x(), p.y())
            if uv is not None:
                u, v = uv
                anchor_u, anchor_v = self._drag_anchor_tex_uv

                # keep anchor texUV under cursor
                self._center_u = float(anchor_u - (u - 0.5) / self._zoom)
                self._center_v = float(anchor_v - (v - 0.5) / self._zoom)
                self._clamp_center()
                self.update()  # redraw with new center

        # --- HOVER readout (works even while dragging) ---
        xy = self._widget_pos_to_image_xy(p.x(), p.y())

        if xy is None or self._display_frame is None:
            self.mousePixelChanged.emit(-1, -1, -1)
            return

        ix, iy = xy
        try:
            val = int(self._display_frame[iy, ix])
        except Exception:
            val = -1

        self.mousePixelChanged.emit(ix, iy, val)

    def wheelEvent(self, event):
        if self._img_w <= 0 or self._img_h <= 0:
            return
        p = event.position().toPoint()
        uv = self._widget_pos_to_a_uv(p.x(), p.y())
        if uv is None:
            return
        u, v = uv

        steps = event.angleDelta().y() / 120.0
        if steps == 0:
            return
        factor = 1.15 ** steps

        old = self._zoom
        new = float(max(self._min_zoom, min(self._max_zoom, old * factor)))
        if new == old:
            return

        # Keep texUV under cursor fixed
        # center = center + (a_uv - 0.5) * (1/old - 1/new)
        self._center_u = float(self._center_u + (u - 0.5) * (1.0 / old - 1.0 / new))
        self._center_v = float(self._center_v + (v - 0.5) * (1.0 / old - 1.0 / new))
        self._zoom = new
        self._clamp_center()
        self.update()

    def set_levels(self, black_u16: int, white_u16: int, gamma: float = 1.0):
        self.black_u16 = int(np.clip(black_u16, 0, 65535))
        self.white_u16 = int(np.clip(white_u16, 0, 65535))
        if self.white_u16 <= self.black_u16:
            self.white_u16 = min(65535, self.black_u16 + 1)
        self.gamma = float(max(gamma, 1e-6))
        self.update()

    def auto_levels(self):
        black = self._display_frame.min()
        white = self._display_frame.max()
        self.set_levels(black, white)
        return black, white

    def set_frame(self, frame_u16: np.ndarray, token=None):
        # If a pending frame hasn’t been uploaded yet, discard it (drop frame)
        if self._have_new and self._pending_token is not None:
            self.frameDiscarded.emit(self._pending_token)

        self._pending = np.ascontiguousarray(frame_u16)
        self._pending_token = token
        self._display_frame = self._pending
        self._display_token = self._pending_token
        self._have_new = True
        self.update()  # schedule paintGL

    def initializeGL(self):
        GL.glDisable(GL.GL_DEPTH_TEST)
        GL.glPixelStorei(GL.GL_UNPACK_ALIGNMENT, 1)

        # Shader program
        self._prog = shaders.compileProgram(
            shaders.compileShader(VERT_SRC, GL.GL_VERTEX_SHADER),
            shaders.compileShader(FRAG_SRC, GL.GL_FRAGMENT_SHADER),
        )
        GL.glUseProgram(self._prog)

        # Fullscreen quad (triangle strip): positions + UVs
        # UVs are set so image is NOT vertically flipped (OpenGL tex origin is bottom-left).
        verts = np.array([
            #  x,    y,    u,   v
            -1.0, -1.0, 0.0, 0.0,
            1.0, -1.0, 1.0, 0.0,
            -1.0, 1.0, 0.0, 1.0,
            1.0, 1.0, 1.0, 1.0,
        ], dtype=np.float32)

        self._vao = GL.glGenVertexArrays(1)
        self._vbo = GL.glGenBuffers(1)
        GL.glBindVertexArray(self._vao)

        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self._vbo)
        GL.glBufferData(GL.GL_ARRAY_BUFFER, verts.nbytes, verts, GL.GL_STATIC_DRAW)

        stride = 4 * 4  # 4 floats per vertex * 4 bytes
        GL.glEnableVertexAttribArray(0)
        GL.glVertexAttribPointer(0, 2, GL.GL_FLOAT, False, stride, ctypes.c_void_p(0))
        GL.glEnableVertexAttribArray(1)
        GL.glVertexAttribPointer(1, 2, GL.GL_FLOAT, False, stride, ctypes.c_void_p(8))

        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, 0)
        GL.glBindVertexArray(0)

        # Texture (allocated when first frame arrives)
        self._tex = GL.glGenTextures(1)
        GL.glBindTexture(GL.GL_TEXTURE_2D, self._tex)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MIN_FILTER, GL.GL_LINEAR)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MAG_FILTER, GL.GL_LINEAR)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_S, GL.GL_CLAMP_TO_EDGE)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_T, GL.GL_CLAMP_TO_EDGE)

        # sampler binding
        loc = GL.glGetUniformLocation(self._prog, "u_tex")
        GL.glUniform1i(loc, 0)

        if self.use_pbo:
            self._pbo = GL.glGenBuffers(2)

    def _ensure_texture_and_pbo(self, w, h):
        if (w, h) == (self._img_w, self._img_h) and self._img_w > 0:
            return

        self._img_w, self._img_h = w, h

        # Allocate texture storage once (then update via TexSubImage)
        GL.glBindTexture(GL.GL_TEXTURE_2D, self._tex)
        GL.glTexImage2D(
            GL.GL_TEXTURE_2D, 0,
            GL.GL_R16, w, h, 0,
            GL.GL_RED, GL.GL_UNSIGNED_SHORT, None
        )

        if self.use_pbo:
            nbytes = w * h * 2
            if nbytes != self._pbo_nbytes:
                self._pbo_nbytes = nbytes
                for i in range(2):
                    GL.glBindBuffer(GL.GL_PIXEL_UNPACK_BUFFER, self._pbo[i])
                    GL.glBufferData(GL.GL_PIXEL_UNPACK_BUFFER, nbytes, None, GL.GL_STREAM_DRAW)
                GL.glBindBuffer(GL.GL_PIXEL_UNPACK_BUFFER, 0)

    def _upload_frame(self, frame_u16: np.ndarray):
        h, w = frame_u16.shape
        self._ensure_texture_and_pbo(w, h)

        GL.glActiveTexture(GL.GL_TEXTURE0)
        GL.glBindTexture(GL.GL_TEXTURE_2D, self._tex)

        if not self.use_pbo:
            GL.glTexSubImage2D(
                GL.GL_TEXTURE_2D, 0, 0, 0, w, h,
                GL.GL_RED, GL.GL_UNSIGNED_SHORT, frame_u16
            )
            return

        # Double PBO ping-pong
        pbo_id = self._pbo[self._pbo_idx]
        self._pbo_idx = 1 - self._pbo_idx

        GL.glBindBuffer(GL.GL_PIXEL_UNPACK_BUFFER, pbo_id)
        # Copy CPU -> PBO (still a copy, but driver can pipeline better)
        GL.glBufferSubData(GL.GL_PIXEL_UNPACK_BUFFER, 0, frame_u16.nbytes, frame_u16)

        # Texture update from PBO (last arg is offset into PBO)
        GL.glTexSubImage2D(
            GL.GL_TEXTURE_2D, 0, 0, 0, w, h,
            GL.GL_RED, GL.GL_UNSIGNED_SHORT, ctypes.c_void_p(0)
        )
        GL.glBindBuffer(GL.GL_PIXEL_UNPACK_BUFFER, 0)

    def _set_uniforms(self):
        # Levels in normalized [0..1]
        b = self.black_u16 / 65535.0
        w = self.white_u16 / 65535.0
        g = self.gamma

        GL.glUseProgram(self._prog)
        GL.glUniform1f(GL.glGetUniformLocation(self._prog, "u_black"), b)
        GL.glUniform1f(GL.glGetUniformLocation(self._prog, "u_white"), w)
        GL.glUniform1f(GL.glGetUniformLocation(self._prog, "u_gamma"), g)

        # Aspect-fit scaling for quad
        if self._img_w > 0 and self._img_h > 0 and self.height() > 0:
            img_aspect = self._img_w / self._img_h
            win_aspect = self.width() / self.height()
            if img_aspect >= win_aspect:
                sx, sy = 1.0, win_aspect / img_aspect
            else:
                sx, sy = img_aspect / win_aspect, 1.0
        else:
            sx, sy = 1.0, 1.0

        self._sx, self._sy = sx, sy
        GL.glUniform2f(GL.glGetUniformLocation(self._prog, "u_scale"), sx, sy)
        GL.glUniform1f(GL.glGetUniformLocation(self._prog, "u_zoom"), float(self._zoom))
        GL.glUniform2f(GL.glGetUniformLocation(self._prog, "u_center"), float(self._center_u), float(self._center_v))

    def paintGL(self):
        GL.glViewport(0, 0, self.width(), self.height())
        GL.glClear(GL.GL_COLOR_BUFFER_BIT)

        if self._have_new and self._pending is not None:
            self._upload_frame(self._pending)

            if self._pending_token is not None:
                self.frameConsumed.emit(self._pending_token)

            self._have_new = False
            self._pending = None
            self._pending_token = None

        if self._img_w == 0:
            return

        self._set_uniforms()

        GL.glBindVertexArray(self._vao)
        GL.glDrawArrays(GL.GL_TRIANGLE_STRIP, 0, 4)
        GL.glBindVertexArray(0)
