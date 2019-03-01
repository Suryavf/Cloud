#version 330 core

// Interpolated values from the vertex shaders
in vec3 fragmentColor;
in vec2 fragmentCoordinate;

// Ouput data
out vec3 color;

float random (in vec2 _st) {
    return fract(sin(dot(_st.xy,
                         vec2(12.9898,78.233)))*
        43758.5453123);
}

// Based on Morgan McGuire @morgan3d
// https://www.shadertoy.com/view/4dS3Wd
float noise (in vec2 _st) {
    vec2 i = floor(_st);
    vec2 f = fract(_st);

    // Four corners in 2D of a tile
    float a = random(i);
    float b = random(i + vec2(1.0, 0.0));
    float c = random(i + vec2(0.0, 1.0));
    float d = random(i + vec2(1.0, 1.0));

    vec2 u = f * f * (3.0 - 2.0 * f);

    return mix(a, b, u.x) +
            (c - a)* u.y * (1.0 - u.x) +
            (d - b) * u.x * u.y;
}

#define NUM_OCTAVES 10

float fbm ( in vec2 _st) {
    float v = 0.0;
    float a = 0.5;
    vec2 shift = vec2(100);
    // Rotate to reduce axial bias

    mat2 rot = mat2( cos(0.5), sin(0.5),
                    -sin(0.5), cos(0.5));
    for (int i = 0; i < NUM_OCTAVES; ++i) {
        v += a * noise(_st);
        _st = rot * _st * 2.0 + shift;
        a *= 0.5;
    }
    return v;
}

void main() {
    vec2 st = gl_FragCoord.xy/vec2(1024.0,768.0);
	//vec2 st = gl_FragCoord.xy/vec2(1024.0,768.0);//u_resolution.xy;
    st.x *= 1024.0f/768.0f;//u_resolution.x/u_resolution.y;


	//fragmentCoordinate.x = fragmentCoordinate.x/36.0;
	//fragmentCoordinate.y = fragmentCoordinate.y/36.0;
	vec2 coord = fragmentCoordinate /50.0;

    vec3 rawcolor = vec3(0.0);
	rawcolor += fbm(coord*5.0);
	
    //gl_FragColor = vec4(color,1.0);
	color =rawcolor;// vec3(0.5, 0.0 ,0.0 );//rawcolor;//rawcolor;
}