////////////////////////////////////////////////////////////////////////
//
//   Harvard University
//   CS175 : Computer Graphics
//   Professor Steven Gortler
//
////////////////////////////////////////////////////////////////////////

#include <vector>
#include <list>
#include <string>
#include <memory>
#include <stdexcept>
#if __GNUG__
#   include <tr1/memory>
#endif

#include <GL/glew.h>
#ifdef __MAC__
#   include <GLUT/glut.h>
#else
#   include <GL/glut.h>
#endif

#include "cvec.h"
#include "matrix4.h"
#include "geometrymaker.h"
#include "ppm.h"
#include "glsupport.h"
#include "quat.h"
#include "rigtform.h"
#include "arcball.h"

#include "asstcommon.h"
#include "scenegraph.h"
#include "drawer.h"
#include "picker.h"
#include "sgutils.h"
#include<iostream>

#include<fstream>

using namespace std;      // for string, vector, iostream, and other standard C++ stuff
using namespace tr1; // for shared_ptr

// G L O B A L S ///////////////////////////////////////////////////

// --------- IMPORTANT --------------------------------------------------------
// Before you start working on this assignment, set the following variable
// properly to indicate whether you want to use OpenGL 2.x with GLSL 1.0 or
// OpenGL 3.x+ with GLSL 1.3.
//
// Set g_Gl2Compatible = true to use GLSL 1.0 and g_Gl2Compatible = false to
// use GLSL 1.3. Make sure that your machine supports the version of GLSL you
// are using. In particular, on Mac OS X currently there is no way of using
// OpenGL 3.x with GLSL 1.3 when GLUT is used.
//
// If g_Gl2Compatible=true, shaders with -gl2 suffix will be loaded.
// If g_Gl2Compatible=false, shaders with -gl3 suffix will be loaded.
// To complete the assignment you only need to edit the shader files that get
// loaded
// ----------------------------------------------------------------------------
const bool g_Gl2Compatible = false;


static const float g_frustMinFov = 60.0;  // A minimal of 60 degree field of view
static float g_frustFovY = g_frustMinFov; // FOV in y direction (updated by updateFrustFovY)

static const float g_frustNear = -0.1;    // near plane
static const float g_frustFar = -50.0;    // far plane
static const float g_groundY = -2.0;      // y coordinate of the ground
static const float g_groundSize = 10.0;   // half the ground length

static int g_windowWidth = 512;
static int g_windowHeight = 512;
static bool g_mouseClickDown = false;    // is the mouse button pressed
static bool g_mouseLClickButton, g_mouseRClickButton, g_mouseMClickButton;
static int g_mouseClickX, g_mouseClickY; // coordinates for mouse click event
static int g_activeShader = 0;


static const int PICKING_SHADER = 2; // index of the picking shader is g_shaerFiles
static const int g_numShaders = 3; // 3 shaders instead of 2
static const char * const g_shaderFiles[g_numShaders][2] = {
  {"./shaders/basic-gl3.vshader", "./shaders/diffuse-gl3.fshader"},
  {"./shaders/basic-gl3.vshader", "./shaders/solid-gl3.fshader"},
  { "./shaders/basic-gl3.vshader", "./shaders/pick-gl3.fshader" }
};
static const char * const g_shaderFilesGl2[g_numShaders][2] = {
  {"./shaders/basic-gl2.vshader", "./shaders/diffuse-gl2.fshader"},
  {"./shaders/basic-gl2.vshader", "./shaders/solid-gl2.fshader"},
  { "./shaders/basic-gl2.vshader", "./shaders/pick-gl2.fshader" }
};
static vector<shared_ptr<ShaderState> > g_shaderStates; // our global shader states

// --------- Geometry

// Macro used to obtain relative offset of a field within a struct
#define FIELD_OFFSET(StructType, field) &(((StructType *)0)->field)

// A vertex with floating point position and normal
struct VertexPN {
  Cvec3f p, n;

  VertexPN() {}
  VertexPN(float x, float y, float z,
           float nx, float ny, float nz)
    : p(x,y,z), n(nx, ny, nz)
  {}

  // Define copy constructor and assignment operator from GenericVertex so we can
  // use make* functions from geometrymaker.h
  VertexPN(const GenericVertex& v) {
    *this = v;
  }

  VertexPN& operator = (const GenericVertex& v) {
    p = v.pos;
    n = v.normal;
    return *this;
  }
};

struct Geometry {
  GlBufferObject vbo, ibo;
  int vboLen, iboLen;

  Geometry(VertexPN *vtx, unsigned short *idx, int vboLen, int iboLen) {
    this->vboLen = vboLen;
    this->iboLen = iboLen;

    // Now create the VBO and IBO
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferData(GL_ARRAY_BUFFER, sizeof(VertexPN) * vboLen, vtx, GL_STATIC_DRAW);

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ibo);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(unsigned short) * iboLen, idx, GL_STATIC_DRAW);
  }

  void draw(const ShaderState& curSS) {
    // Enable the attributes used by our shader
    safe_glEnableVertexAttribArray(curSS.h_aPosition);
    safe_glEnableVertexAttribArray(curSS.h_aNormal);

    // bind vbo
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    safe_glVertexAttribPointer(curSS.h_aPosition, 3, GL_FLOAT, GL_FALSE, sizeof(VertexPN), FIELD_OFFSET(VertexPN, p));
    safe_glVertexAttribPointer(curSS.h_aNormal, 3, GL_FLOAT, GL_FALSE, sizeof(VertexPN), FIELD_OFFSET(VertexPN, n));

    // bind ibo
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ibo);

    // draw!
    glDrawElements(GL_TRIANGLES, iboLen, GL_UNSIGNED_SHORT, 0);

    // Disable the attributes used by our shader
    safe_glDisableVertexAttribArray(curSS.h_aPosition);
    safe_glDisableVertexAttribArray(curSS.h_aNormal);
  }
};

typedef SgGeometryShapeNode<Geometry> MyShapeNode;

// Vertex buffer and index buffer associated with the ground and cube geometry
static shared_ptr<Geometry> g_ground, g_cube, g_sphere;

static shared_ptr<SgRootNode> g_world;
static shared_ptr<SgRbtNode> g_skyNode, g_groundNode, g_robot1Node, g_robot2Node;
static shared_ptr<SgRbtNode> g_currentPickedRbtNode; // used later when you do picking
static shared_ptr<SgRbtNode> g_currentView;

static bool g_pick = false;
// --------- Scene

static const Cvec3 g_light1(2.0, 3.0, 14.0), g_light2(-2, -3.0, -5.0);  // define two lights positions in world space
//static Matrix4 g_skyRbt = Matrix4::makeTranslation(Cvec3(0.0, 0.25, 4.0));
//static Matrix4 g_objectRbt[2] = { Matrix4::makeTranslation(Cvec3(-1, 0, 0)), Matrix4::makeTranslation(Cvec3(1, 0, 0)) };  // currently only 1 obj is defined

static RigTForm g_ballRbt = RigTForm(Cvec3(0, 0, 0)); 
static Cvec3f g_ballColor = Cvec3f(1, 1, 0) ;

static int index_view, index_mani, index_w_s;
static double g_arcballScreenRadius = 128.0 , g_arcballScale = 1.0;

list < vector <RigTForm> > frame_list;
list < vector <RigTForm> >::iterator now_iter;
static int current_frame_num = -1;
static char file_name[] = "animation.txt"; 
static int g_msBetweenKeyFrames = 2000; // 2 seconds between keyframes 
static int g_animateFramesPerSecond = 60; // frames to render per second during animation playback
static bool g_playing = false;
static double g_ani_t = 0;
///////////////// END OF G L O B A L S //////////////////////////////////////////////////




static void initGround() {
  // A x-z plane at y = g_groundY of dimension [-g_groundSize, g_groundSize]^2
  VertexPN vtx[4] = {
    VertexPN(-g_groundSize, g_groundY, -g_groundSize, 0, 1, 0),
    VertexPN(-g_groundSize, g_groundY,  g_groundSize, 0, 1, 0),
    VertexPN( g_groundSize, g_groundY,  g_groundSize, 0, 1, 0),
    VertexPN( g_groundSize, g_groundY, -g_groundSize, 0, 1, 0),
  };
  unsigned short idx[] = {0, 1, 2, 0, 2, 3};
  g_ground.reset(new Geometry(&vtx[0], &idx[0], 4, 6));
}

static void initCubes() {
  int ibLen, vbLen;
  getCubeVbIbLen(vbLen, ibLen);

  // Temporary storage for cube geometry
  vector<VertexPN> vtx(vbLen);
  vector<unsigned short> idx(ibLen);

  makeCube(1, vtx.begin(), idx.begin());
  g_cube.reset(new Geometry(&vtx[0], &idx[0], vbLen, ibLen));
  //makeCube(2, vtx.begin(), idx.begin());
  //g_cube.reset(new Geometry(&vtx[0], &idx[0], vbLen, ibLen));

}

static void initSphere() {
	int ibLen, vbLen;
	getSphereVbIbLen(20, 20, vbLen, ibLen);

	// Temporary storage for cube geometry
	vector<VertexPN> vtx(vbLen);
	vector<unsigned short> idx(ibLen);

	//cout << 0.25*min(g_windowWidth, g_windowHeight) << endl;
	//makeSphere(0.0025*min(g_windowWidth, g_windowHeight), 20, 20, vtx.begin(), idx.begin());
	makeSphere(1, 20, 20, vtx.begin(), idx.begin());
	g_sphere.reset(new Geometry(&vtx[0], &idx[0], vbLen, ibLen));

}

// takes a projection matrix and send to the the shaders
static void sendProjectionMatrix(const ShaderState& curSS, const Matrix4& projMatrix) {
  GLfloat glmatrix[16];
  projMatrix.writeToColumnMajorMatrix(glmatrix); // send projection matrix
  safe_glUniformMatrix4fv(curSS.h_uProjMatrix, glmatrix);
}


// update g_frustFovY from g_frustMinFov, g_windowWidth, and g_windowHeight
static void updateFrustFovY() {
  if (g_windowWidth >= g_windowHeight)
    g_frustFovY = g_frustMinFov;
  else {
    const double RAD_PER_DEG = 0.5 * CS175_PI/180;
    g_frustFovY = atan2(sin(g_frustMinFov * RAD_PER_DEG) * g_windowHeight / g_windowWidth, cos(g_frustMinFov * RAD_PER_DEG)) / RAD_PER_DEG;
  }
}

static Matrix4 makeProjectionMatrix() {
  return Matrix4::makeProjection(
           g_frustFovY, g_windowWidth / static_cast <double> (g_windowHeight),
           g_frustNear, g_frustFar);
}

static void drawStuff(const ShaderState& curSS, bool picking) {


  // build & send proj. matrix to vshader
  const Matrix4 projmat = makeProjectionMatrix();
  sendProjectionMatrix(curSS, projmat);

  // use the skyRbt as the eyeRbt
  RigTForm eyeRbt;

  eyeRbt = getPathAccumRbt(g_world, g_currentView);

  const RigTForm invEyeRbt = inv(eyeRbt);

  const Cvec3 eyeLight1 = Cvec3(invEyeRbt * Cvec4(g_light1, 1)); // g_light1 position in eye coordinates
  const Cvec3 eyeLight2 = Cvec3(invEyeRbt * Cvec4(g_light2, 1)); // g_light2 position in eye coordinates
  safe_glUniform3f(curSS.h_uLight, eyeLight1[0], eyeLight1[1], eyeLight1[2]);
  safe_glUniform3f(curSS.h_uLight2, eyeLight2[0], eyeLight2[1], eyeLight2[2]);


  if (!picking) {
	  Drawer drawer(invEyeRbt, curSS);
	  g_world->accept(drawer);

	  // draw arcball as part of asst3
	  RigTForm sphRbt;
	  bool cnt = false;
	  if (g_currentPickedRbtNode == g_skyNode) {
		  if (index_w_s == 0 && index_view == 0) {
			  sphRbt = g_ballRbt;
			  cnt = true;
		  }
	  }
	  else {
		  if (g_currentView != g_currentPickedRbtNode) {
			  sphRbt = getPathAccumRbt(g_world, g_currentPickedRbtNode);
			  cnt = true;
		  }
	  }
	  if (cnt) {
		  RigTForm MVM = invEyeRbt * sphRbt;

		  if (!g_mouseMClickButton && !(g_mouseLClickButton && g_mouseRClickButton)) {
			  g_arcballScale = getScreenToEyeScale(MVM.getTranslation()[2], g_frustFovY, g_windowHeight);
		  }
		  const Matrix4 sc = Matrix4::makeScale(g_arcballScale * g_arcballScreenRadius);

		  Matrix4 NMVM = normalMatrix(rigTFormToMatrix(MVM) * sc);
		  sendModelViewNormalMatrix(curSS, rigTFormToMatrix(MVM) * sc, NMVM);
		  safe_glUniform3f(curSS.h_uColor, g_ballColor[0], g_ballColor[1], g_ballColor[2]);

		  glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);

		  g_sphere->draw(curSS);
		  glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
	  }
  }
  else {
	  Picker picker(invEyeRbt, curSS);
	  g_world->accept(picker);
	  glFlush();
	  g_currentPickedRbtNode = picker.getRbtNodeAtXY(g_mouseClickX, g_mouseClickY);
	  if (g_currentPickedRbtNode == NULL) g_currentPickedRbtNode = g_skyNode;
	  if (g_currentPickedRbtNode == g_groundNode)
		  g_currentPickedRbtNode = g_skyNode;
  }
}

static void display() {
  glUseProgram(g_shaderStates[g_activeShader]->program);
  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);                   // clear framebuffer color&depth

  drawStuff(*g_shaderStates[g_activeShader], g_pick);

                        
  if (!g_pick) {
	  glutSwapBuffers();         // show the back buffer (where we rendered stuff)
  }
  checkGlErrors();
}

static void reshape(const int w, const int h) {
  g_windowWidth = w;
  g_windowHeight = h;
  glViewport(0, 0, w, h);

  g_arcballScreenRadius = 0.25 * min(g_windowHeight, g_windowWidth);

  cerr << "Size of window is now " << w << "x" << h << endl;
  updateFrustFovY();
  glutPostRedisplay();
}

static RigTForm arc(const int x, const int y) {

	const double x1 = (double) g_mouseClickX, y1 = (double) g_mouseClickY;

	const double x2 = (double)x, y2 = (double)g_windowHeight - y - 1;



	RigTForm eyeRbt;

	eyeRbt = getPathAccumRbt(g_world, g_currentView);

	RigTForm sphRbt;
	if (g_currentPickedRbtNode == g_skyNode) {
		if (index_w_s == 0 && index_view == 0) {
			sphRbt = g_ballRbt;
		}
	}
	else {
		if (g_currentView != g_currentPickedRbtNode) {
			sphRbt = getPathAccumRbt(g_world, g_currentPickedRbtNode);
		}
	}

	Cvec2 sph_xy;

	sph_xy = getScreenSpaceCoord((inv(eyeRbt) * sphRbt).getTranslation(), makeProjectionMatrix(),
		g_frustNear, g_frustFovY, g_windowWidth, g_windowHeight);

	Cvec3 v1 = normalize(Cvec3(x1 - sph_xy[0], y1 - sph_xy[1], sqrt(max(0.0, pow(g_arcballScreenRadius, 2) - pow(x1 - sph_xy[0], 2) - pow(y1 - sph_xy[1], 2)))));
	Cvec3 v2 = normalize(Cvec3(x2 - sph_xy[0], y2 - sph_xy[1], sqrt(max(0.0, pow(g_arcballScreenRadius, 2) - pow(x2 - sph_xy[0], 2) - pow(y2 - sph_xy[1], 2)))));

	return RigTForm(Quat(0, v2) * Quat(0, v1* -1));


}
static void motion(const int x, const int y) {
	
  //if (index_mani == 0 && index_view != 0) return;
  const double dx = x - g_mouseClickX;
  const double dy = g_windowHeight - y - 1 - g_mouseClickY;

  bool cnt = false;


  if (g_currentPickedRbtNode == g_skyNode) {
	  if (index_w_s == 0 && index_view == 0) {
		  cnt = true;
	  }
  }
  else {
	  if (g_currentView != g_currentPickedRbtNode) cnt = true;
  }
  RigTForm m;

  if (g_mouseLClickButton && !g_mouseRClickButton) { // left button down?
	  if (!cnt)
		  m = RigTForm(Quat::makeXRotation(-dy) * Quat::makeYRotation(dx));
	  else
		  m = arc(x, y);
  }
  else if (g_mouseRClickButton && !g_mouseLClickButton) { // right button down?
	  if(!cnt)
		  m = RigTForm(Cvec3(dx, dy, 0) * 0.01);
	  else
		  m = RigTForm(Cvec3(dx, dy, 0) * g_arcballScale);
  }
  else if (g_mouseMClickButton || (g_mouseLClickButton && g_mouseRClickButton)) {  // middle or (left and right) button down?
	  if(!cnt)
		  m = RigTForm(Cvec3(0, 0, -dy) * 0.01);
	  else
		  m = RigTForm(Cvec3(0, 0, -dy) * g_arcballScale);
  }
  RigTForm eyeRbt;

  eyeRbt = getPathAccumRbt(g_world, g_currentView);

  RigTForm A;

  if (g_currentPickedRbtNode == g_skyNode) {
	  if (index_w_s == 0 && index_view == 0) {
		  A = linFact(eyeRbt);
	  }
	  else {
		  A = eyeRbt;
	  }
  }
  else {
	A = inv(getPathAccumRbt(g_world, g_currentPickedRbtNode, 1)) * transFact(getPathAccumRbt(g_world, g_currentPickedRbtNode)) * linFact(eyeRbt);
  }
  /*for (int i = 0; i < 4; i++){
	  for (int j = 0; j < 4; j++){
		  cout << eyeRbt(i, j) << " ";
	  }
	  cout << endl;
  }*/
  RigTForm invA = inv(A);
  //cout << index_mani << " " << index_view << " " << cnt << endl;
  if (g_mouseClickDown) {
	  if (g_currentPickedRbtNode == g_skyNode) {
		  if (index_w_s == 0 && index_view == 0) {
			  g_skyNode->setRbt(A * inv(m) * invA * g_skyNode->getRbt());
		  }
		  else {
			  m = transFact(m)*inv(linFact(m));
			  g_currentView->setRbt(A * m * invA * g_currentView->getRbt());
		  }
	  }
	  else {
		  if (g_currentView != g_currentPickedRbtNode) g_currentPickedRbtNode->setRbt(A * m * invA * g_currentPickedRbtNode->getRbt());
		  else {
			  g_currentPickedRbtNode->setRbt(A * inv(m) * invA * g_currentPickedRbtNode->getRbt());
		  }
	  }
    glutPostRedisplay(); // we always redraw if we changed the scene
  }

  g_mouseClickX = x;
  g_mouseClickY = g_windowHeight - y - 1;


}
static void pick() {
	// We need to set the clear color to black, for pick rendering.
	// so let's save the clear color
	GLdouble clearColor[4];
	glGetDoublev(GL_COLOR_CLEAR_VALUE, clearColor);

	glClearColor(0, 0, 0, 0);

	// using PICKING_SHADER as the shader
	glUseProgram(g_shaderStates[PICKING_SHADER]->program);

	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	drawStuff(*g_shaderStates[PICKING_SHADER], true);

	// Uncomment below and comment out the glutPostRedisplay in mouse(...) call back
	// to see result of the pick rendering pass
	 //glutSwapBuffers();

	//Now set back the clear color
	glClearColor(clearColor[0], clearColor[1], clearColor[2], clearColor[3]);

	checkGlErrors();
}

static void mouse(const int button, const int state, const int x, const int y) {
  g_mouseClickX = x;
  g_mouseClickY = g_windowHeight - y - 1;  // conversion from GLUT window-coordinate-system to OpenGL window-coordinate-system

  g_mouseLClickButton |= (button == GLUT_LEFT_BUTTON && state == GLUT_DOWN);
  g_mouseRClickButton |= (button == GLUT_RIGHT_BUTTON && state == GLUT_DOWN);
  g_mouseMClickButton |= (button == GLUT_MIDDLE_BUTTON && state == GLUT_DOWN);

  g_mouseLClickButton &= !(button == GLUT_LEFT_BUTTON && state == GLUT_UP);
  g_mouseRClickButton &= !(button == GLUT_RIGHT_BUTTON && state == GLUT_UP);
  g_mouseMClickButton &= !(button == GLUT_MIDDLE_BUTTON && state == GLUT_UP);

  g_mouseClickDown = g_mouseLClickButton || g_mouseRClickButton || g_mouseMClickButton;

  if (g_pick && g_mouseLClickButton && !g_mouseRClickButton) {
	  pick();
	  g_pick = false;
	  cout << "Picking mode is off" << endl;
  }
  
  glutPostRedisplay();
}
static void show_current_frame() {
	if (frame_list.size() != 0) {
		vector < shared_ptr<SgRbtNode> > rbt_nodes;
		dumpSgRbtNodes(g_world, rbt_nodes);
		for (int i = 0; i < rbt_nodes.size(); i++) {
			rbt_nodes[i]->setRbt( (*now_iter)[i] );
		}
	}
}
static void create_new_frame() {
	vector <RigTForm> frame;
	vector < shared_ptr<SgRbtNode> > rbt_nodes;
	dumpSgRbtNodes(g_world, rbt_nodes);

	for (int i = 0; i < rbt_nodes.size(); i++) {
		frame.push_back(rbt_nodes[i]->getRbt());
	}
	if (frame_list.size() == 0) {
		frame_list.push_back(frame);
		now_iter = frame_list.begin();
	}
	else {
		now_iter++;
		frame_list.insert(now_iter, frame);
		now_iter--;
	}

	current_frame_num++;

	cout << "Create new frame [" << current_frame_num << "]" << endl;
}

static void update_current_frame() {
	if (frame_list.size() == 0) {
		create_new_frame();
	}
	else {
		vector <RigTForm> frame;
		vector < shared_ptr<SgRbtNode> > rbt_nodes;
		dumpSgRbtNodes(g_world, rbt_nodes);

		for (int i = 0; i < rbt_nodes.size(); i++) {
			frame.push_back(rbt_nodes[i]->getRbt());
		}
		*now_iter = frame;
	}
	cout << "Copying scend graph to current frame [" << current_frame_num << "]" << endl;

}

static void delete_current_frame() {
	if (frame_list.size() == 0) {
		cout << "Frame list is now EMPTY" << endl;
	}
	else {
		list < vector <RigTForm> >::iterator imsi = now_iter;
		cout << "Deleting current frame [" << current_frame_num << "]" << endl;
		if (now_iter == frame_list.begin()) {
			imsi++;
			frame_list.erase(now_iter);
			now_iter = imsi;
		}
		else {
			imsi--;
			current_frame_num--;
			frame_list.erase(now_iter);
			now_iter = imsi;
		}

		if (frame_list.size() == 0) {
			current_frame_num = -1;
			cout << "No frames defined" << endl;
		}
		else {
			show_current_frame();
			cout << "Now at frame [" << current_frame_num << "]" << endl;
		}
	}
}
static void to_next_frame() {
	if (frame_list.size() != 0) {
		now_iter++;
		if (now_iter != frame_list.end()) {
			current_frame_num++; 
			cout << "Stepped forward to frame [" << current_frame_num << "]" << endl;
			show_current_frame();
		}
		else now_iter--;
	}
}
static void to_prev_frame() {
	if (frame_list.size() != 0) {
		if (now_iter != frame_list.begin()) {
			now_iter--;
			current_frame_num--;
			cout << "Stepped backward to frame [" << current_frame_num << "]" << endl;
			show_current_frame();
		}
	}
}
static void read_from_file() {
	frame_list.clear();
	int num_frame, num_rbt, i, j;
	double w, x, y, z;
	ifstream f(file_name, ios::binary);
	f >> num_frame >> num_rbt;	Cvec3 t;	Quat r;
	vector <RigTForm> frame;	for (i = 0; i < num_frame; i++) {		frame.clear();		for (j = 0; j < num_rbt; j++) {			f >> x >> y >> z;			t = Cvec3(x, y, z);			f >> w >> x >> y >> z;			r = Quat(w, x, y, z);			frame.push_back(RigTForm(t, r));		}		frame_list.push_back(frame);	}	cout << "Reading animation from animation.txt" << endl;
	cout << num_frame << " " << "frames read." << endl;	if (frame_list.size() == 0) {		current_frame_num = -1;	}	else {		current_frame_num = 0;
		now_iter = frame_list.begin();
		show_current_frame();		cout << "Now at frame[0]" << endl;	}	f.close();
}
static void write_to_file() {
	ofstream f(file_name, ios::binary);
	vector < shared_ptr<SgRbtNode> > rbt_nodes;
	dumpSgRbtNodes(g_world, rbt_nodes);	f << frame_list.size() << " " << rbt_nodes.size() << endl;
	list < vector <RigTForm> >::iterator iter;	int i;	RigTForm imsi;	Cvec3 t;	Quat r;	for (iter = frame_list.begin(); iter != frame_list.end(); iter++) {		for (i = 0; i < (*iter).size(); i++) {			imsi = (*iter)[i];			t = imsi.getTranslation();			r = imsi.getRotation();			f << t[0] << " " << t[1] << " " << t[2] << " " << r[0] << " " << r[1] << " " << r[2] << " " << r[3] << endl;		}	}	f.close();	cout << "Writing animation to animation.txt" << endl;
}
void interpolate(double a) {
	list < vector <RigTForm> >::iterator first = now_iter;
	list < vector <RigTForm> >::iterator second = now_iter;
	second++;
	vector < shared_ptr<SgRbtNode> > rbt_nodes;
	dumpSgRbtNodes(g_world, rbt_nodes);
	RigTForm imsi, imsi2;

	for (int i = 0; i < rbt_nodes.size(); i++) {
		imsi = (*first)[i];
		imsi2 = (*second)[i];
		rbt_nodes[i]->setRbt(RigTForm(RigTForm::lerp(imsi.getTranslation(), imsi2.getTranslation(), a),
									  RigTForm::slerp(imsi.getRotation(), imsi2.getRotation(), a)));
	}

}
bool interpolateAndDisplay(double t) {
	int first = 1 + floor(t);
	if (first > current_frame_num) {
		current_frame_num++;
		now_iter++;
		/*cout << "ms: " << t*g_msBetweenKeyFrames << endl;
		cout << "t: " << t << endl;
		cout << current_frame_num << endl;*/
	}else if (first < current_frame_num) {
		current_frame_num--;
		now_iter--;
		/*cout << "ms: " << t*g_msBetweenKeyFrames << endl;
		cout << "t: " << t << endl;
		cout << current_frame_num << endl;*/
	}
	if (current_frame_num < frame_list.size() - 2 && g_playing) {
		double a = t - floor(t);
		interpolate(a);
		glutPostRedisplay();
		return false;
	}
	else {
		g_playing = false;
		return true;
	}
}
static void animateTimerCallback(int ms) {
	//double t = (1000 / g_animateFramesPerSecond) / (double)g_msBetweenKeyFrames;
	double t = (double)ms / (double)g_msBetweenKeyFrames;
	//g_ani_t += t;
	//bool endReached = interpolateAndDisplay(g_ani_t);
	bool endReached = interpolateAndDisplay(t);
	if (!endReached) {
		glutTimerFunc(1000 / g_animateFramesPerSecond, animateTimerCallback, ms + 1000 / g_animateFramesPerSecond);
	}else {
		cout << "Finished playing animation" << endl;
		while (current_frame_num < frame_list.size() - 2) {
			current_frame_num++;
			now_iter++;
		}
		show_current_frame();
		glutPostRedisplay();
		cout << "Now at frame [" << current_frame_num << "]" << endl;

	}
}
static void anime() {
	if (!g_playing) {
		if (frame_list.size() < 4) {
			cout << "Cannot play animation with less than 4 keyframes." << endl;
			return;
		}
		g_playing = true;
		cout << "Playing animation..." << endl;
		now_iter = frame_list.begin();
		now_iter++;
		current_frame_num = 1;
		show_current_frame();
		//g_ani_t = 0;
		animateTimerCallback(0);
		
	}
	else {
		cout << "Stopping animation..." << endl;
		g_playing = false;
	}
}
static void keyboard(const unsigned char key, const int x, const int y) {
  switch (key) {
  case 27:
    exit(0);                                  // ESC
  case 'h':
    cout << " ============== H E L P ==============\n\n"
    << "h\t\thelp menu\n"
    << "s\t\tsave screenshot\n"
    << "f\t\tToggle flat shading on/off.\n"
    << "o\t\tCycle object to edit\n"
    << "v\t\tCycle view\n"
    << "drag left mouse to rotate\n" << endl;
    break;
  case 's':
    glFlush();
    writePpmScreenshot(g_windowWidth, g_windowHeight, "out.ppm");
    break;
  case 'f':
    g_activeShader ^= 1;
    break;
  case 'v':
	  index_view = (index_view + 1) % 3; 
	  if (index_view == 0) {
		  g_currentView = g_skyNode;
	  }
	  else if (index_view == 1) {
		  g_currentView = g_robot1Node;
	  }
	  else {
		  g_currentView = g_robot2Node;
	  }
	  glutPostRedisplay();
	  break;
  case 'p':
	  g_pick = true;
	  cout << "Picking mode is on" << endl;
	  glutPostRedisplay();
	  break;
  case 'm':
	  index_w_s = (index_w_s + 1) % 2;
	  glutPostRedisplay();
	  break;

  case 'n':
	  if (g_playing)
		  cout << "Cannot operate when playing animation" << endl;
	  else {
		  create_new_frame();
		  glutPostRedisplay();
	  }
	  break;
  case 32:
	  if (g_playing)
		  cout << "Cannot operate when playing animation" << endl;
	  else {
		  show_current_frame();
		  if (frame_list.size() == 0)  cout << "No key frame defined" << endl;
		  else cout << "Loading current key frame [" << current_frame_num << "] to scend graph" << endl;
		  glutPostRedisplay();
	  }
	  break;
  case 'u':
	  if (g_playing)
		  cout << "Cannot operate when playing animation" << endl;
	  else {
		  update_current_frame();
		  glutPostRedisplay();
	  }
	  break;
  case '>':
	  if (g_playing)
		  cout << "Cannot operate when playing animation" << endl;
	  else {
		  to_next_frame();
		  glutPostRedisplay();
	  }
	  break;
  case '<':
	  if (g_playing)
		  cout << "Cannot operate when playing animation" << endl;
	  else {
		  to_prev_frame();
		  glutPostRedisplay();
	  }
	  break;
  case 'd':
	  if (g_playing)
		  cout << "Cannot operate when playing animation" << endl;
	  else {
		  delete_current_frame();
		  glutPostRedisplay();
	  }
	  break;
  case 'w':
	  write_to_file();
	  glutPostRedisplay();
	  break;
  case 'i':
	  if (g_playing)
		  cout << "Cannot operate when playing animation" << endl;
	  else {
		  read_from_file();
		  glutPostRedisplay();
	  }
	  break;
  case '+':
	  g_msBetweenKeyFrames = max(100, g_msBetweenKeyFrames - 100);
	  cout << g_msBetweenKeyFrames << " ms between keyframes." << endl;
	  break;
  case '-':
	  g_msBetweenKeyFrames = min(10000, g_msBetweenKeyFrames + 100);
	  cout << g_msBetweenKeyFrames << " ms between keyframes." << endl;
	  break;
  case 'y':
	  anime();
	  break;
  }
  glutPostRedisplay();
}

static void initGlutState(int argc, char * argv[]) {
  glutInit(&argc, argv);                                  // initialize Glut based on cmd-line args
  glutInitDisplayMode(GLUT_RGBA|GLUT_DOUBLE|GLUT_DEPTH);  //  RGBA pixel channels and double buffering
  glutInitWindowSize(g_windowWidth, g_windowHeight);      // create a window
  glutCreateWindow("Assignment 5");                       // title the window

  glutDisplayFunc(display);                               // display rendering callback
  glutReshapeFunc(reshape);                               // window reshape callback
  glutMotionFunc(motion);                                 // mouse movement callback
  glutMouseFunc(mouse);                                   // mouse click callback
  glutKeyboardFunc(keyboard);
}

static void initGLState() {
  glClearColor(128./255., 200./255., 255./255., 0.);
  glClearDepth(0.);
  glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
  glPixelStorei(GL_PACK_ALIGNMENT, 1);
  glCullFace(GL_BACK);
  glEnable(GL_CULL_FACE);
  glEnable(GL_DEPTH_TEST);
  glDepthFunc(GL_GREATER);
  glReadBuffer(GL_BACK);
  if (!g_Gl2Compatible)
    glEnable(GL_FRAMEBUFFER_SRGB);
}

static void initShaders() {
  g_shaderStates.resize(g_numShaders);
  for (int i = 0; i < g_numShaders; ++i) {
    if (g_Gl2Compatible)
      g_shaderStates[i].reset(new ShaderState(g_shaderFilesGl2[i][0], g_shaderFilesGl2[i][1]));
    else
      g_shaderStates[i].reset(new ShaderState(g_shaderFiles[i][0], g_shaderFiles[i][1]));
  }
}

static void initGeometry() {
  initGround();
  initCubes();
  initSphere();
}

static void constructRobot(shared_ptr<SgTransformNode> base, const Cvec3& color) {

	const float ARM_LEN = 0.7,
		ARM_THICK = 0.25,
		TORSO_LEN = 1.5,
		TORSO_THICK = 0.25,
		TORSO_WIDTH = 1,
		LEG_LEN = 1,
		LEG_THICK = 0.25,
		HEAD_RAD = 0.35;
	const int NUM_JOINTS = 10,
		NUM_SHAPES = 10;

	struct JointDesc {
		int parent;
		float x, y, z;
	};

	JointDesc jointDesc[NUM_JOINTS] = {
		{ -1 }, // torso
		{ 0,  TORSO_WIDTH / 2, TORSO_LEN / 2, 0 }, // upper right arm
		{ 1,  ARM_LEN, 0, 0 }, // lower right arm
		{ 0,  -TORSO_WIDTH / 2, TORSO_LEN / 2, 0 }, // upper left arm
		{ 3,  -ARM_LEN, 0, 0 }, // lower left arm
		{ 0,  TORSO_WIDTH / 2 - LEG_THICK / 2, -TORSO_LEN / 2, 0 }, // upper right leg
		{ 5,  0, -LEG_LEN, 0 }, // lower right leg
		{ 0,  -TORSO_WIDTH / 2 + LEG_THICK / 2, -TORSO_LEN / 2, 0 }, // upper left leg
		{ 7,  0, -LEG_LEN, 0 }, // lower left leg
		{ 0,  0, TORSO_LEN / 2, 0 } // head
		
	};

	struct ShapeDesc {
		int parentJointId;
		float x, y, z, sx, sy, sz;
		shared_ptr<Geometry> geometry;
	};

	ShapeDesc shapeDesc[NUM_SHAPES] = {
		{ 0, 0,         0, 0, TORSO_WIDTH, TORSO_LEN, TORSO_THICK, g_cube }, // torso
		{ 1, ARM_LEN / 2, 0, 0, ARM_LEN / 2, ARM_THICK / 2, ARM_THICK / 2, g_sphere }, // upper right arm
		{ 2, ARM_LEN / 2, 0, 0, ARM_LEN, ARM_THICK, ARM_THICK, g_cube }, // lower right arm
		{ 3, -ARM_LEN / 2, 0, 0, ARM_LEN / 2, ARM_THICK / 2, ARM_THICK / 2, g_sphere }, // upper left arm
		{ 4, -ARM_LEN / 2, 0, 0, ARM_LEN, ARM_THICK, ARM_THICK, g_cube }, // lower left arm
		{ 5, 0, -LEG_LEN / 2, 0, LEG_THICK / 2, LEG_LEN / 2, LEG_THICK / 2, g_sphere }, // upper right leg
		{ 6, 0, -LEG_LEN / 2, 0, LEG_THICK, LEG_LEN, LEG_THICK, g_cube }, // lower right leg
		{ 7, 0, -LEG_LEN / 2, 0, LEG_THICK / 2, LEG_LEN / 2, LEG_THICK / 2, g_sphere }, // upper left leg
		{ 8, 0, -LEG_LEN / 2, 0, LEG_THICK, LEG_LEN, LEG_THICK, g_cube }, // lower left leg
		{ 9, 0, 0.5, 0, HEAD_RAD, HEAD_RAD, HEAD_RAD, g_sphere }  // head
	};

	shared_ptr<SgTransformNode> jointNodes[NUM_JOINTS];

	for (int i = 0; i < NUM_JOINTS; ++i) {
		if (jointDesc[i].parent == -1)
			jointNodes[i] = base;
		else {
			jointNodes[i].reset(new SgRbtNode(RigTForm(Cvec3(jointDesc[i].x, jointDesc[i].y, jointDesc[i].z))));
			jointNodes[jointDesc[i].parent]->addChild(jointNodes[i]);
		}
	}
	for (int i = 0; i < NUM_SHAPES; ++i) {
		shared_ptr<MyShapeNode> shape(
			new MyShapeNode(shapeDesc[i].geometry,
				color,
				Cvec3(shapeDesc[i].x, shapeDesc[i].y, shapeDesc[i].z),
				Cvec3(0, 0, 0),
				Cvec3(shapeDesc[i].sx, shapeDesc[i].sy, shapeDesc[i].sz)));
		jointNodes[shapeDesc[i].parentJointId]->addChild(shape);
	}
}

static void initScene() {
	g_world.reset(new SgRootNode());

	g_skyNode.reset(new SgRbtNode(RigTForm(Cvec3(0.0, 0.25, 4.0))));
	g_currentView = g_skyNode;
	g_currentPickedRbtNode = g_skyNode;

	g_groundNode.reset(new SgRbtNode());
	g_groundNode->addChild(shared_ptr<MyShapeNode>(
		new MyShapeNode(g_ground, Cvec3(0.1, 0.95, 0.1))));

	g_robot1Node.reset(new SgRbtNode(RigTForm(Cvec3(-2, 1, 0))));
	g_robot2Node.reset(new SgRbtNode(RigTForm(Cvec3(2, 1, 0))));

	constructRobot(g_robot1Node, Cvec3(1, 0, 0)); // a Red robot
	constructRobot(g_robot2Node, Cvec3(0, 0, 1)); // a Blue robot

	g_world->addChild(g_skyNode);
	g_world->addChild(g_groundNode);
	g_world->addChild(g_robot1Node);
	g_world->addChild(g_robot2Node);
}

int main(int argc, char * argv[]) {
  try {
    initGlutState(argc,argv);

    glewInit(); // load the OpenGL extensions

    cout << (g_Gl2Compatible ? "Will use OpenGL 2.x / GLSL 1.0" : "Will use OpenGL 3.x / GLSL 1.3") << endl;
    if ((!g_Gl2Compatible) && !GLEW_VERSION_3_0)
      throw runtime_error("Error: card/driver does not support OpenGL Shading Language v1.3");
    else if (g_Gl2Compatible && !GLEW_VERSION_2_0)
      throw runtime_error("Error: card/driver does not support OpenGL Shading Language v1.0");

    initGLState();
    initShaders();
    initGeometry();
	initScene();

    glutMainLoop();
    return 0;
  }
  catch (const runtime_error& e) {
    cout << "Exception caught: " << e.what() << endl;
    return -1;
  }
}
