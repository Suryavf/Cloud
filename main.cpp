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

int main(void){

	// We have to call cudaGLSetGLDevice if we want to use OpenGL interoperability.
    cudaGLSetGLDevice(0);

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
/*
	// Projection matrix : 45� Field of View, 4:3 ratio, display range : 0.1 unit <-> 100 units
	glm::mat4 Projection = glm::perspective(glm::radians(45.0f), 4.0f / 3.0f, 0.1f, 100.0f);
	
	// Camera matrix
	glm::mat4 View = glm::lookAt(
								glm::vec3(4,3,3), // Camera is at (4,3,3), in World Space
								glm::vec3(0,0,0), // and looks at the origin
								glm::vec3(0,1,0)  // Head is up (set to 0,-1,0 to look upside-down)
						   );
	// Model matrix : an identity matrix (model will be at the origin)
	glm::mat4 Model      = glm::mat4(1.0f);

	// Our ModelViewProjection : multiplication of our 3 matrices
	glm::mat4 MVP        = Projection * View * Model; // Remember, matrix multiplication is the other way around
*/

/*    
 *  Create Sphere. 3D vertex
 *  ========================
 */
	Sphere sphr = Sphere(1.0f, 36, 18, true);
	
	GLfloat *gColorBufferData = new GLfloat[ sphr.getInterleavedVertexSize() ];
	for(size_t i=0; i<sphr.getInterleavedVertexSize(); ++i)
		gColorBufferData[i] = 0.5f;

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
 *  OpenGL Loop!
 *  ============
 */
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

	// Close OpenGL window and terminate GLFW
	glfwTerminate();

	return 0;
}