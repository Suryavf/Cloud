// Include standard headers
#include <stdio.h>
#include <stdlib.h>
#include <iostream>

// Include GLEW
#include <GL/glew.h>

// Include GLFW
#include <GLFW/glfw3.h>
GLFWwindow* window;

// CUDA headers
#include <cuda_runtime_api.h>
#include <cuda_gl_interop.h>

// Include GLM
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
using namespace glm;

// Include Common
#include "common/shader.hpp"
#include "common/texture.hpp"
#include "common/controls.hpp"
#include "sphere.h"

#include <iostream>
#include <chrono>

class Timer
{
public:
    Timer() : beg_(clock_::now()) {}
    void reset() { beg_ = clock_::now(); }
    double elapsed() const { 
        return std::chrono::duration_cast<second_>
            (clock_::now() - beg_).count(); }

private:
    typedef std::chrono::high_resolution_clock clock_;
    typedef std::chrono::duration<double, std::ratio<1> > second_;
    std::chrono::time_point<clock_> beg_;
};



// Textures 
GLuint texOpticalDepthID;
cudaGraphicsResource *cudaOpticalDepthResource;
cudaArray            *cudaOpticalDepthArray   ;
dim3 texOpticalDepthDim(32, 16, 32*16);

GLuint   texScatteringID;
cudaGraphicsResource *cudaScatteringResource;
cudaArray            *cudaScatteringArray   ;
dim3 texScatteringDim(32, 64, 16);

int main(void){

/*    
 *  Initialise GLFW
 *  ===============
 */
	if( !glfwInit() ){
		fprintf( stderr, "Failed to initialize GLFW\n" );
		getchar();
		return -1;
	}

    // GLFW configuration
    glfwWindowHint(GLFW_SAMPLES, 4);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
	glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE); // To make MacOS happy; should not be needed
	glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

    // Open a window and create its OpenGL context
	window = glfwCreateWindow( 1024, 768, "Realistic Cloud", NULL, NULL);
	if( window == NULL ){
		fprintf( stderr, "Failed to open GLFW window. If you have an Intel GPU, they are not 3.3 compatible. Try the 2.1 version of the tutorials.\n" );
		getchar();
		glfwTerminate();
		return -1;
	}

    // Make window
    glfwMakeContextCurrent(window);

    // Initialize GLEW
	glewExperimental = true; // Needed for core profile
	if (glewInit() != GLEW_OK) {
		fprintf(stderr, "Failed to initialize GLEW\n");
		getchar();
		glfwTerminate();
		return -1;
	}

	// Ensure we can capture the escape key being pressed below
	glfwSetInputMode(window, GLFW_STICKY_KEYS, GL_TRUE);

/*    
 *  Basic Setting OpenGL
 *  ====================
 */
    // Dark background
	glClearColor(0.0f, 0.0f, 0.0f, 0.0f);

	// Enable depth test
	glEnable(GL_DEPTH_TEST);

	// Accept fragment if it closer to the camera than the former one
	glDepthFunc(GL_LESS); 

/*    
 *  Basic Setting CUDA
 *  ==================
 */
	int device = 0;
	cudaSetDevice    ( device );
	cudaGLSetGLDevice( device ); // We have to call cudaGLSetGLDevice if we want to use OpenGL interoperability.

/*    
 *  VAO: Vertex Array Object
 *  ========================
 */
	GLuint VAO;
	glGenVertexArrays(1,&VAO);
	glBindVertexArray(VAO);

/*    
 *  Load shaders: vertex and fragment
 *  =================================
 */
	// Create and compile our GLSL program from the shaders
	GLuint programID = LoadShaders(   "../vertexshader.glsl", 
	                                "../fragmentshader.glsl");


/*    
 *  MVP: Modelo Vista Proyección
 *  ============================
 */
	// Get a handle for our "MVP" uniform
	GLuint MatrixID = glGetUniformLocation(programID, "MVP");
	GLuint   TimeID = glGetUniformLocation(programID, "Time");

/*    
 *  Create Sphere. 3D vertex
 *  ========================
 */
	Sphere sphr = Sphere(1.0f, 32, 16, true);

/*    
 *  Buffer to OpenGL
 *  ================
 */
	GLuint vertexbuffer;
	glGenBuffers(1, &vertexbuffer);
	glBindBuffer(GL_ARRAY_BUFFER, vertexbuffer);
	glBufferData(GL_ARRAY_BUFFER, sphr.getGLVertexSize(), sphr.getGLvertices(), GL_STATIC_DRAW);

	GLuint normalbuffer;
	glGenBuffers(1, &normalbuffer);
	glBindBuffer(GL_ARRAY_BUFFER, normalbuffer);
	glBufferData(GL_ARRAY_BUFFER, sphr.getGLnormalSize(), sphr.getGLnormals(), GL_STATIC_DRAW);

	GLuint coordinatesbuffer;
	glGenBuffers(1, &coordinatesbuffer);
	glBindBuffer(GL_ARRAY_BUFFER, coordinatesbuffer);
	glBufferData(GL_ARRAY_BUFFER, sphr.getGLcoordinateSize(), sphr.getGLcoordinates(), GL_STATIC_DRAW);


/*    
 *  Create textures
 *  ===============
 */
	// Texture optical depth
	glGenTextures(1, &texOpticalDepthID);
	glBindTexture(GL_TEXTURE_3D, texOpticalDepthID);
	{	
	/*
		El parámetro GL_NEAREST especifica que no debe producirse ningún filtrado; simplemente 
		devuelva el píxel cerrado a la coordenada de textura solicitada.
	 */

		glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MIN_FILTER, GL_NEAREST        );
		glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MAG_FILTER, GL_NEAREST        );
		glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_S,     GL_CLAMP_TO_BORDER);
		glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_T,     GL_CLAMP_TO_BORDER);
		glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_T,     GL_CLAMP_TO_BORDER);
	/*
	  	Dado que las coordenadas de la textura generalmente se definen en el rango [0..1), 
		el acceso a un píxel fuera de este rango generalmente resultará en un error (como 
		intentar acceder a una matriz fuera de los límites) pero la configuración GL_CLAMP_TO_EDGE 
		nos permite solicitar un píxel de la textura fuera del rango normalizado sin acceder 
		a la memoria fuera de los límites. Las coordenadas de la textura simplemente se fijarán 
		en el rango permitido cuando se acceda a la textura.
	 */

		// Allocate memory for the texture
		glTexImage3D(GL_TEXTURE_3D, 0, GL_RGBA32F, texOpticalDepthDim.x, texOpticalDepthDim.y, texOpticalDepthDim.z, 0, GL_RGBA, GL_FLOAT, NULL);
	}
	// Unbind the texture
	glBindTexture(GL_TEXTURE_3D, 0);

	// Texture scattering
	glGenTextures(1, &texScatteringID);
	glBindTexture(GL_TEXTURE_3D, texScatteringID);
	{
		glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MIN_FILTER, GL_NEAREST        );
		glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MAG_FILTER, GL_NEAREST        );
		glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_S,     GL_CLAMP_TO_BORDER);
		glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_T,     GL_CLAMP_TO_BORDER);
		glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_T,     GL_CLAMP_TO_BORDER);

		// Allocate memory for the texture
		glTexImage3D(GL_TEXTURE_3D, 0, GL_RGBA32F, texScatteringDim.x, texScatteringDim.y, texScatteringDim.z, 0, GL_RGBA, GL_FLOAT, NULL);
	}
	// Unbind the texture
	glBindTexture(GL_TEXTURE_3D, 0);


/*    
 *  OpenGL-CUDA connection
 *  ======================
 */
	// Optical Depth
	cudaGraphicsGLRegisterImage(&cudaOpticalDepthResource, texOpticalDepthID, GL_TEXTURE_3D, 
	                             cudaGraphicsRegisterFlagsSurfaceLoadStore); // Register Image (texture) to CUDA Resource
	cudaGraphicsMapResources(1, &cudaOpticalDepthResource, 0); // Map CUDA resource
	
	// Scattering
	cudaGraphicsGLRegisterImage(&cudaScatteringResource, texScatteringID, GL_TEXTURE_3D, 
	                             cudaGraphicsRegisterFlagsSurfaceLoadStore); // Register Image (texture) to CUDA Resource
	cudaGraphicsMapResources(1, &cudaScatteringResource, 0); // Map CUDA resource

/*    
 *  CUDA execution
 *  ==============
 */
	// Get Optical Depth 
	cudaGraphicsSubResourceGetMappedArray(&cudaOpticalDepthArray, cudaOpticalDepthResource, 0, 0); //Get mapped array
	//launch_kernel(cudaOpticalDepthArray, texOpticalDepthDim);
	
	
	// Get Scattering
	cudaGraphicsSubResourceGetMappedArray(&cudaScatteringArray, cudaScatteringResource, 0, 0); //Get mapped array
	//launch_kernel(cudaScatteringArray, texScatteringDim);
	

/*    
 *  OpenGL-CUDA disconnection
 *  =========================
 */
	cudaGraphicsUnmapResources(1, &cudaOpticalDepthResource, 0);
	cudaGraphicsUnregisterResource(cudaOpticalDepthResource);

	cudaGraphicsUnmapResources(1, &cudaScatteringResource, 0);
	cudaGraphicsUnregisterResource(cudaScatteringResource);
	

/*    
 *  OpenGL Loop!
 *  ============
 */
	Timer tmr;
	float time;
	do{
	  /*
	   °Basic operations
	    ****************/
		// Clear the screen
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

		// Use our shader
		glUseProgram(programID);


	  /*
	   °MVP: Modelo Vista Proyección
		****************************/
		// Compute the MVP matrix from keyboard and mouse input
		computeMatricesFromInputs();
		glm::mat4 ProjectionMatrix = getProjectionMatrix();
		glm::mat4 ViewMatrix = getViewMatrix();
		glm::mat4 ModelMatrix = glm::mat4(1.0);
		glm::mat4 MVP = ProjectionMatrix * ViewMatrix * ModelMatrix;

		// Send our transformation to the currently bound shader, 
		// in the "MVP" uniform
		glUniformMatrix4fv(MatrixID, 1, GL_FALSE, &MVP[0][0]);


	  /*
	   °Draw mesh
		*********/
		// 1rst attribute buffer : vertices
		glEnableVertexAttribArray(0);
		glBindBuffer(GL_ARRAY_BUFFER, vertexbuffer);
		glVertexAttribPointer(
			0,                  // attribute. No particular reason for 0, but must match the layout in the shader.
			3,                  // size
			GL_FLOAT,           // type
			GL_FALSE,           // normalized?
			0,                  // stride
			(void*)0            // array buffer offset
		);

		// 2nd attribute buffer : normal
		glEnableVertexAttribArray(1);
		glBindBuffer(GL_ARRAY_BUFFER, normalbuffer);
		glVertexAttribPointer(
			1,                                // attribute. No particular reason for 1, but must match the layout in the shader.
			3,                                // size 
			GL_FLOAT,                         // type
			GL_FALSE,                         // normalized?
			0,                                // stride
			(void*)0                          // array buffer offset
		);

		// 3nd attribute buffer : coordinate
		glEnableVertexAttribArray(1);
		glBindBuffer(GL_ARRAY_BUFFER, coordinatesbuffer);
		glVertexAttribPointer(
			2,                                // attribute. No particular reason for 1, but must match the layout in the shader.
			2,                                // size 
			GL_FLOAT,                         // type
			GL_FALSE,                         // normalized?
			0,                                // stride
			(void*)0                          // array buffer offset
		);
		
		// Pass time
		time = float(tmr.elapsed());
		glUniform1f(TimeID,time);

		// Draw the triangle !
		glDrawArrays(GL_TRIANGLES, 0, sphr.getGLVertexCount()); // 12*3 indices starting at 0 -> 12 triangles
		
		glDisableVertexAttribArray(0);
		glDisableVertexAttribArray(1);
		glDisableVertexAttribArray(2);

		// Swap buffers
		glfwSwapBuffers(window);
		glfwPollEvents();
	}
	while( glfwGetKey(window, GLFW_KEY_ESCAPE ) != GLFW_PRESS &&
		   glfwWindowShouldClose(window) == 0 );


/*    
 *  Cleanup VBO and shader
 *  ======================
 */
	glDeleteBuffers(1, &     vertexbuffer);
	glDeleteBuffers(1, &     normalbuffer);
	glDeleteBuffers(1, &coordinatesbuffer);
	glDeleteProgram(programID);
	glDeleteVertexArrays(1, &VAO);

	glDeleteTextures(1, &texOpticalDepthID);
	glDeleteTextures(1, &texScatteringID);

	cudaDeviceReset();

	// Close OpenGL window and terminate GLFW
	glfwTerminate();

	return 0;
}