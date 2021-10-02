varying float v_border_distance;

#ifdef VERTEX_SHADER
attribute vec3 a_pos;
attribute float a_border_distance;
uniform mat4 u_projection_matrix;
uniform mat4 u_view_matrix;
uniform mat4 u_model_matrix;
void main() {
    v_border_distance = a_border_distance;
    gl_Position = u_projection_matrix * u_view_matrix * u_model_matrix * vec4(a_pos, 1.0);
}
#endif

#ifdef FRAGMENT_SHADER
uniform vec4 u_border_color;
uniform vec4 u_color;
void main() {
    if (v_border_distance < 0.1) {
        gl_FragColor = u_border_color;
    } else {
        gl_FragColor = u_color;
    }
    if (gl_FragColor.w < 0.5) {
        discard;
    }
}
#endif