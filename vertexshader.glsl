#version 330 core

// Input vertex data, different for all executions of this shader.
layout(location = 0) in vec3 vertexPosition_modelspace;
layout(location = 1) in vec3 normal;
layout(location = 2) in vec2 coordinate;

// Output data ; will be interpolated for each fragment.
out vec3 fragmentColor;
out vec2 fragmentCoordinate;
out vec3 fragmentNoise;

// Values that stay constant for the whole mesh.
uniform  mat4 MVP ;
uniform float time;

// Pseudorandom number generator
// https://www.geeks3d.com/20100831/shader-library-noise-and-pseudo-random-number-generator-in-glsl/
float rnd(vec2 x){
    int n = int(x.x * 40.0 + x.y * 6400.0);
    n = (n << 13) ^ n;
    return 1.0 - float( (n * (n * n * 15731 + 789221) + \
             1376312589) & 0x7fffffff) / 1073741824.0;
}

/*
 *  Main Funtion
 *  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
 */
void main(){	

	// Output position of the vertex, in clip space : MVP * position
	gl_Position =  MVP * vec4(vertexPosition_modelspace,1);

	// The color of each vertex will be interpolated
	// to produce the color of each fragment
	fragmentColor = vec3(0.5,0.5,0.5);//vertexColor;
	fragmentCoordinate = coordinate;

    // Noise
    float n = rnd( gl_Vertex.xyz );
    vec3  f3Noise = vec3(n,n,n);
    fRndAzimuthBias = f3Noise.y+(f3Noise.x-0.5)*time*5e-2;
}

 