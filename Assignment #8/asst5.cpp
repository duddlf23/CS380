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
#include "geometry.h"

#include "mesh.h"

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


/*static const int PICKING_SHADER = 2; // index of the picking shader is g_shaerFiles
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
static vector<shared_ptr<ShaderState> > g_shaderStates; // our global shader states*/

// --------- Materials
// This should replace all the contents in the Shaders section, e.g., g_numShaders, g_shaderFiles, and so on
static shared_ptr<Material> g_redDiffuseMat,
g_blueDiffuseMat,
g_bumpFloorMat,
g_arcballMat,
g_pickingMat,
g_lightMat,
g_specularMat;

shared_ptr<Material> g_overridingMaterial;

// --------- Geometry
typedef SgGeometryShapeNode MyShapeNode;

static shared_ptr<SimpleGeometryPN> g_mesh;
static Mesh mesh1, mesh2;
// Vertex buffer and index buffer associated with the ground and cube geometry
static shared_ptr<Geometry> g_ground, g_cube, g_sphere, g_light1, g_light2;

static shared_ptr<SgRootNode> g_world;
static shared_ptr<SgRbtNode> g_skyNode, g_groundNode, g_robot1Node, g_robot2Node;
static shared_ptr<SgRbtNode> g_currentPickedRbtNode; // used later when you do picking
static shared_ptr<SgRbtNode> g_currentView;


static shared_ptr<SgRbtNode> g_light1Node, g_light2Node;
static shared_ptr<SgRbtNode> g_meshNode;

static bool g_pick = false;
// --------- Scene

static const Cvec3 light1_pos(-2.0, 3.0, 4.0), light2_pos(4.0, 3.0, -3.0);  // define two lights positions in world space
static const Cvec3 meshpos(0, 0, 0);
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

static float g_animate_mesh_speed = 400.0;
static bool smooth_flat = false;
static int g_subdiv_step = 0;
///////////////// END OF G L O B A L S //////////////////////////////////////////////////




static void initGround() {
	int ibLen, vbLen;
	getPlaneVbIbLen(vbLen, ibLen);

	// Temporary storage for cube Geometry
	vector<VertexPNTBX> vtx(vbLen);
	vector<unsigned short> idx(ibLen);

	makePlane(g_groundSize * 2, vtx.begin(), idx.begin());
	g_ground.reset(new SimpleIndexedGeometryPNTBX(&vtx[0], &idx[0], vbLen, ibLen));
}

static void initCubes() {
	int ibLen, vbLen;
	getCubeVbIbLen(vbLen, ibLen);

	// Temporary storage for cube Geometry
	vector<VertexPNTBX> vtx(vbLen);
	vector<unsigned short> idx(ibLen);

	makeCube(1, vtx.begin(), idx.begin());
	g_cube.reset(new SimpleIndexedGeometryPNTBX(&vtx[0], &idx[0], vbLen, ibLen));
}

static void initSphere() {
	int ibLen, vbLen;
	getSphereVbIbLen(20, 10, vbLen, ibLen);

	// Temporary storage for sphere Geometry
	vector<VertexPNTBX> vtx(vbLen);
	vector<unsigned short> idx(ibLen);
	makeSphere(1, 20, 10, vtx.begin(), idx.begin());
	g_sphere.reset(new SimpleIndexedGeometryPNTBX(&vtx[0], &idx[0], vtx.size(), idx.size()));
}
static void up_nor_1() {
	int k = mesh1.getNumVertices();

	int i;

	for (i = 0; i < k; i++) {
		Mesh::Vertex imsi = mesh1.getVertex(i);
		imsi.setNormal(Cvec3(0, 0, 0));
	}

	for (i = 0; i < k; i++) {
		Mesh::VertexIterator imsi_iter(mesh1.getVertex(i).getIterator());
		Mesh::VertexIterator imsi_iter2(imsi_iter);

		Cvec3 sum = imsi_iter.getFace().getNormal();
		++imsi_iter;

		while (imsi_iter != imsi_iter2) {
			sum += imsi_iter.getFace().getNormal();
			++imsi_iter;
		}

		if (dot(sum, sum) > CS175_EPS2) {
			sum.normalize();
		}

		mesh1.getVertex(i).setNormal(sum);

	}
}
static void up_nor_2() {
	int k = mesh2.getNumVertices();

	int i;

	for (i = 0; i < k; i++) {
		Mesh::Vertex imsi = mesh2.getVertex(i);
		imsi.setNormal(Cvec3(0, 0, 0));
	}

	for (i = 0; i < k; i++) {
		Mesh::VertexIterator imsi_iter(mesh2.getVertex(i).getIterator());
		Mesh::VertexIterator imsi_iter2(imsi_iter);

		Cvec3 sum = imsi_iter.getFace().getNormal();
		++imsi_iter;

		while (imsi_iter != imsi_iter2) {
			sum += imsi_iter.getFace().getNormal();
			++imsi_iter;
		}

		if (dot(sum, sum) > CS175_EPS2) {
			sum.normalize();
		}

		mesh2.getVertex(i).setNormal(sum);

	}
}
static void face_sub() {
	int i, j, k = mesh2.getNumFaces();
	
	for (i = 0; i < k; i++) {
		Mesh::Face imsi = mesh2.getFace(i);

		Cvec3 sum(0.0, 0.0, 0.0);

		for (j = 0; j < imsi.getNumVertices(); j++) {
			sum += imsi.getVertex(j).getPosition();
		}

		sum = sum * (1.0 / (double)imsi.getNumVertices());

		mesh2.setNewFaceVertex(imsi, sum);
	}

}
static void edge_sub() {
	int i, j, k = mesh2.getNumEdges();

	for (i = 0; i < k; i++) {
		Mesh::Edge imsi = mesh2.getEdge(i);

		Cvec3 sum(0.0, 0.0, 0.0);

		sum = sum + imsi.getVertex(0).getPosition();
		sum = sum + imsi.getVertex(1).getPosition();
		sum = sum + mesh2.getNewFaceVertex(imsi.getFace(0));
		sum = sum + mesh2.getNewFaceVertex(imsi.getFace(1));

		sum = sum * 0.25;

		mesh2.setNewEdgeVertex(imsi, sum);
	}


}
static void vert_sub() {
	int i, j, k = mesh2.getNumVertices();

	for (i = 0; i < k; i++) {
		Mesh::Vertex imsi = mesh2.getVertex(i);

		Mesh::VertexIterator imsi_iter(imsi.getIterator());
		Mesh::VertexIterator imsi_iter2(imsi_iter);

		int n = 1;
		Cvec3 sum1(0.0, 0.0, 0.0), sum2(0.0, 0.0, 0.0);

		sum1 = sum1 + imsi_iter.getVertex().getPosition();
		sum2 = sum2 + mesh2.getNewFaceVertex(imsi_iter.getFace());
		++imsi_iter;

		while (imsi_iter != imsi_iter2) {
			sum1 = sum1 + imsi_iter.getVertex().getPosition();
			sum2 = sum2 + mesh2.getNewFaceVertex(imsi_iter.getFace());
			n++, ++imsi_iter;
		}

		Cvec3 ans = imsi.getPosition() * ((double)(n - 2) / n);
		ans = ans + sum1 * (1.0 / (n * n)) + sum2 * (1.0 / (n * n));

		mesh2.setNewVertexVertex(imsi, ans);
	}

}
void mesh_disp(float t) {

	mesh2 = Mesh(mesh1);

	int i, k = mesh2.getNumVertices();

	for (i = 0; i < k; i++) {
		Cvec3 imsi = mesh2.getVertex(i).getPosition();
		mesh2.getVertex(i).setPosition(imsi + imsi * 0.75 * (sin(i + t / 100)));

	}
	for (i = 0; i < g_subdiv_step; i++) {
		face_sub();
		edge_sub();
		vert_sub();

		mesh2.subdivide();
	}
	up_nor_2();

	vector<VertexPN> v;

	k = mesh2.getNumFaces();

	for (i = 0; i < k; i++) {
		Mesh::Face imsi = mesh2.getFace(i);

		for (int j = 1; j < imsi.getNumVertices() - 1; j++) {
			if (smooth_flat) {
				Mesh::Vertex imsi2 = imsi.getVertex(0);
				v.push_back(VertexPN(imsi2.getPosition(), imsi2.getNormal()));
				Mesh::Vertex imsi3 = imsi.getVertex(j);
				v.push_back(VertexPN(imsi3.getPosition(), imsi3.getNormal()));
				Mesh::Vertex imsi4 = imsi.getVertex(j + 1);
				v.push_back(VertexPN(imsi4.getPosition(), imsi4.getNormal()));
			}
			else {
				Mesh::Vertex imsi2 = imsi.getVertex(0);
				v.push_back(VertexPN(imsi2.getPosition(), imsi.getNormal()));
				Mesh::Vertex imsi3 = imsi.getVertex(j);
				v.push_back(VertexPN(imsi3.getPosition(), imsi.getNormal()));
				Mesh::Vertex imsi4 = imsi.getVertex(j + 1);
				v.push_back(VertexPN(imsi4.getPosition(), imsi.getNormal()));
			}
		}
	}
	g_mesh -> upload(&v[0], v.size());
	glutPostRedisplay();
}
static void animate_mesh_callback(int ms) {

	float t = (float)ms / 100;

	mesh_disp(t);
	glutTimerFunc(10, animate_mesh_callback, ms + g_animate_mesh_speed);
}
static void initMesh() {
	string s = "cube.mesh";
	mesh1 = Mesh();
	mesh1.load(s.c_str());
	up_nor_1();

	mesh2 = Mesh(mesh1);

	vector<VertexPN> v;

	int i, j, k = mesh2.getNumFaces();

	for (i = 0; i < k; i++) {
		Mesh::Face imsi = mesh2.getFace(i);

		for (int j = 1; j < imsi.getNumVertices() - 1; j++) {
			if (smooth_flat) {
				Mesh::Vertex imsi2 = imsi.getVertex(0);
				v.push_back(VertexPN(imsi2.getPosition(), imsi2.getNormal()));
				Mesh::Vertex imsi3 = imsi.getVertex(j);
				v.push_back(VertexPN(imsi3.getPosition(), imsi3.getNormal()));
				Mesh::Vertex imsi4 = imsi.getVertex(j + 1);
				v.push_back(VertexPN(imsi4.getPosition(), imsi4.getNormal()));
			}
			else {
				Mesh::Vertex imsi2 = imsi.getVertex(0);
				v.push_back(VertexPN(imsi2.getPosition(), imsi.getNormal()));
				Mesh::Vertex imsi3 = imsi.getVertex(j);
				v.push_back(VertexPN(imsi3.getPosition(), imsi.getNormal()));
				Mesh::Vertex imsi4 = imsi.getVertex(j + 1);
				v.push_back(VertexPN(imsi4.getPosition(), imsi.getNormal()));
			}
		}
	}

	g_mesh.reset(new SimpleGeometryPN());
	g_mesh -> upload(&v[0], v.size());

	animate_mesh_callback(0);

}

// takes a projection matrix and send to the the shaders
inline void sendProjectionMatrix(Uniforms& uniforms, const Matrix4& projMatrix) {
	uniforms.put("uProjMatrix", projMatrix);
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

static void drawStuff(bool picking) {


	// Declare an empty uniforms
	Uniforms uniforms;

  // build & send proj. matrix to vshader
  const Matrix4 projmat = makeProjectionMatrix();
  // send proj. matrix to be stored by uniforms,
  // as opposed to the current vtx shader
  sendProjectionMatrix(uniforms, projmat);

  // use the skyRbt as the eyeRbt
  RigTForm eyeRbt;

  eyeRbt = getPathAccumRbt(g_world, g_currentView);

  const RigTForm invEyeRbt = inv(eyeRbt);
	
  Cvec3 light1 = getPathAccumRbt(g_world, g_light1Node).getTranslation();
  Cvec3 light2 = getPathAccumRbt(g_world, g_light2Node).getTranslation();
																 // send the eye space coordinates of lights to uniforms

  // transform to eye space, and set to uLight uniform 
  
  uniforms.put("uLight", Cvec3(invEyeRbt * Cvec4(light1, 1)));
  uniforms.put("uLight2", Cvec3(invEyeRbt * Cvec4(light2, 1)));


  if (!picking) {
	  // initialize the drawer with our uniforms, as opposed to curSS
	  Drawer drawer(invEyeRbt, uniforms);
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
		  // Use uniforms as opposed to curSS
		  sendModelViewNormalMatrix(uniforms, rigTFormToMatrix(MVM) * sc, normalMatrix(rigTFormToMatrix(MVM) * sc));

		  // No more glPolygonMode calls

		  g_arcballMat->draw(*g_sphere, uniforms);

		  // No more glPolygonMode calls

	  }
  }
  else {
	  // intialize the picker with our uniforms, as opposed to curSS
	  Picker picker(invEyeRbt, uniforms);    
	  
	  // set overiding material to our picking material
	  g_overridingMaterial = g_pickingMat;

	  g_world->accept(picker);
	  // unset the overriding material
	  g_overridingMaterial.reset();
	  glFlush();
	  g_currentPickedRbtNode = picker.getRbtNodeAtXY(g_mouseClickX, g_mouseClickY);
	  if (g_currentPickedRbtNode == NULL) g_currentPickedRbtNode = g_skyNode;
	  if (g_currentPickedRbtNode == g_groundNode)
		  g_currentPickedRbtNode = g_skyNode;
  }
}

static void display() {
	// No more glUseProgram

	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	drawStuff(false); // no more curSS

	glutSwapBuffers();

	checkGlErrors();
}
static void pick() {
	// We need to set the clear color to black, for pick rendering.
	// so let's save the clear color
	GLdouble clearColor[4];
	glGetDoublev(GL_COLOR_CLEAR_VALUE, clearColor);

	glClearColor(0, 0, 0, 0);

	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	// No more glUseProgram
	drawStuff(true); // no more curSS

					 // Uncomment below and comment out the glutPostRedisplay in mouse(...) call back
					 // to see result of the pick rendering pass
					 // glutSwapBuffers();

					 //Now set back the clear color
	glClearColor(clearColor[0], clearColor[1], clearColor[2], clearColor[3]);

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
Cvec3 c_r_inter_t(Cvec3 prev, Cvec3 first, Cvec3 second, Cvec3 after, double a) {
	const Cvec3 d = RigTForm::get_d(prev, first, second);
	const Cvec3 e = RigTForm::get_e(first, second, after);

	const Cvec3 f = RigTForm::lerp(first, d, a);
	const Cvec3 g = RigTForm::lerp(d, e, a);
	const Cvec3 h = RigTForm::lerp(e, second, a);
	const Cvec3 m = RigTForm::lerp(f, g, a);
	const Cvec3 n = RigTForm::lerp(g, h, a);

	const Cvec3 c = RigTForm::lerp(m, n, a);

	return c;
	
}
Quat c_r_inter_r(Quat prev, Quat first, Quat second, Quat after, double a) {
	const Quat d = RigTForm::get_d(prev, first, second);
	const Quat e = RigTForm::get_e(first, second, after);

	const Quat f = RigTForm::slerp(first, d, a);
	const Quat g = RigTForm::slerp(d, e, a);
	const Quat h = RigTForm::slerp(e, second, a);
	const Quat m = RigTForm::slerp(f, g, a);
	const Quat n = RigTForm::slerp(g, h, a);

	const Quat c = RigTForm::slerp(m, n, a);

	return c;

}
void interpolate(double a) {
	list < vector <RigTForm> >::iterator prev = now_iter;
	prev--;
	list < vector <RigTForm> >::iterator first = now_iter;
	list < vector <RigTForm> >::iterator second = now_iter;
	second++;
	list < vector <RigTForm> >::iterator after = second;
	after++;
	vector < shared_ptr<SgRbtNode> > rbt_nodes;
	dumpSgRbtNodes(g_world, rbt_nodes);
	RigTForm imsi0, imsi1, imsi2, imsi3;

	for (int i = 0; i < rbt_nodes.size(); i++) {
		imsi0 = (*prev)[i];
		imsi1 = (*first)[i];
		imsi2 = (*second)[i];
		imsi3 = (*after)[i];
		rbt_nodes[i]->setRbt(RigTForm(c_r_inter_t(imsi0.getTranslation(), imsi1.getTranslation(), imsi2.getTranslation(), imsi3.getTranslation(), a),
									  c_r_inter_r(imsi0.getRotation(), imsi1.getRotation(), imsi2.getRotation(), imsi3.getRotation(), a)));
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

  case 'f':
	  smooth_flat = !smooth_flat;
	  break;

  case '0':
	  if (g_subdiv_step < 7) g_subdiv_step++;
	  cout << "Subdivision levels = " << g_subdiv_step << endl;
	  break;

  case '9':
	  if (g_subdiv_step > 0) g_subdiv_step--;
	  cout << "Subdivision levels = " << g_subdiv_step << endl;
	  break;
  case '7':
	  g_animate_mesh_speed /= 2.0;
	  break;
  case '8':
	  g_animate_mesh_speed *= 2.0;
	  break;
  }
  glutPostRedisplay();
}

static void initGlutState(int argc, char * argv[]) {
  glutInit(&argc, argv);                                  // initialize Glut based on cmd-line args
  glutInitDisplayMode(GLUT_RGBA|GLUT_DOUBLE|GLUT_DEPTH);  //  RGBA pixel channels and double buffering
  glutInitWindowSize(g_windowWidth, g_windowHeight);      // create a window
  glutCreateWindow("Assignment 8");                       // title the window

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

static void initMaterials() {
	// Create some prototype materials
	Material diffuse("./shaders/basic-gl3.vshader", "./shaders/diffuse-gl3.fshader");
	Material solid("./shaders/basic-gl3.vshader", "./shaders/solid-gl3.fshader");
	Material specular("./shaders/basic-gl3.vshader", "./shaders/specular-gl3.fshader");

	// copy diffuse prototype and set red color
	g_redDiffuseMat.reset(new Material(diffuse));
	g_redDiffuseMat->getUniforms().put("uColor", Cvec3f(1, 0, 0));

	// copy diffuse prototype and set blue color
	g_blueDiffuseMat.reset(new Material(diffuse));
	g_blueDiffuseMat->getUniforms().put("uColor", Cvec3f(0, 0, 1));

	// normal mapping material
	g_bumpFloorMat.reset(new Material("./shaders/normal-gl3.vshader", "./shaders/normal-gl3.fshader"));
	g_bumpFloorMat->getUniforms().put("uTexColor", shared_ptr<ImageTexture>(new ImageTexture("Fieldstone.ppm", true)));
	g_bumpFloorMat->getUniforms().put("uTexNormal", shared_ptr<ImageTexture>(new ImageTexture("FieldstoneNormal.ppm", false)));

	// copy solid prototype, and set to wireframed rendering
	g_arcballMat.reset(new Material(solid));
	g_arcballMat->getUniforms().put("uColor", Cvec3f(0.27f, 0.82f, 0.35f));
	g_arcballMat->getRenderStates().polygonMode(GL_FRONT_AND_BACK, GL_LINE);

	// copy solid prototype, and set to color white
	g_lightMat.reset(new Material(solid));
	g_lightMat->getUniforms().put("uColor", Cvec3f(1, 1, 1));

	g_specularMat.reset(new Material(specular));
	g_specularMat->getUniforms().put("uColor", Cvec3f(0.9, 0.9, 0.1));

	// pick shader
	g_pickingMat.reset(new Material("./shaders/basic-gl3.vshader", "./shaders/pick-gl3.fshader"));
};


static void initGeometry() {
  initGround();
  initCubes();
  initSphere();
  initMesh();
}

static void constructRobot(shared_ptr<SgTransformNode> base, shared_ptr<Material> material) {

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
		shared_ptr<SgGeometryShapeNode> shape(
			new MyShapeNode(shapeDesc[i].geometry,
				material, // USE MATERIAL as opposed to color
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
		new MyShapeNode(g_ground, g_bumpFloorMat, Cvec3(0, g_groundY, 0))));

	g_robot1Node.reset(new SgRbtNode(RigTForm(Cvec3(-2, 1, 0))));
	g_robot2Node.reset(new SgRbtNode(RigTForm(Cvec3(2, 1, 0))));


	g_light1Node.reset(new SgRbtNode(RigTForm(light1_pos)));
	g_light1Node->addChild(shared_ptr<MyShapeNode>(new MyShapeNode(g_sphere, g_lightMat, Cvec3(0, 0, 0))));

	g_light2Node.reset(new SgRbtNode(RigTForm(light2_pos)));
	g_light2Node->addChild(shared_ptr<MyShapeNode>(new MyShapeNode(g_sphere, g_lightMat, Cvec3(0, 0, 0))));

	constructRobot(g_robot1Node, g_redDiffuseMat); // a Red robot
	constructRobot(g_robot2Node, g_blueDiffuseMat); // a Blue robot

	g_meshNode.reset(new SgRbtNode(RigTForm(meshpos)));
	g_meshNode->addChild(shared_ptr<MyShapeNode>(new MyShapeNode(g_mesh, g_specularMat, Cvec3(0, 0, 0))));

	g_world->addChild(g_skyNode);
	g_world->addChild(g_groundNode);
	g_world->addChild(g_robot1Node);
	g_world->addChild(g_robot2Node);
	g_world->addChild(g_light1Node);
	g_world->addChild(g_light2Node);
	g_world->addChild(g_meshNode);
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
    //initShaders();
	initMaterials();
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
