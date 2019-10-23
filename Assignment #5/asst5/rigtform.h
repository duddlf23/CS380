#ifndef RIGTFORM_H
#define RIGTFORM_H

#include <iostream>
#include <cassert>

#include "matrix4.h"
#include "quat.h"

class RigTForm {
  Cvec3 t_; // translation component
  Quat r_;  // rotation component represented as a quaternion

public:
  RigTForm() : t_(0) {
    assert(norm2(Quat(1,0,0,0) - r_) < CS175_EPS2);
  }

  RigTForm(const Cvec3& t, const Quat& r) {
    //TODO
	  t_ = t;
	  r_ = r;
  }

  explicit RigTForm(const Cvec3& t) {
    // TODO
	  t_ = t;
	  r_ = Quat();
  }

  explicit RigTForm(const Quat& r) {
    // TODO
	  t_ = Cvec3(0, 0, 0);
	  r_ = r;
  }

  Cvec3 getTranslation() const {
    return t_;
  }

  Quat getRotation() const {
    return r_;
  }

  RigTForm& setTranslation(const Cvec3& t) {
    t_ = t;
    return *this;
  }

  RigTForm& setRotation(const Quat& r) {
    r_ = r;
    return *this;
  }

  Cvec4 operator * (const Cvec4& a) const {
    // TODO
	  //assert(a != Cvec4(0, 0, 0, 0));

	  return this -> r_ * a + Cvec4(this -> t_, 0);
  }

  RigTForm operator * (const RigTForm& a) const {
    // TODO
	  RigTForm A;
	  Cvec4 imsi = this->r_ * Cvec4(a.t_, 1);

	  //A.t_ = this->t_ + Cvec3(imsi[0], imsi[1], imsi[2]);
	  A.t_ = this->t_ + Cvec3(imsi);
	  A.r_ = this->r_ * a.r_;

	  //Cvec3(a[0], a[1], a[2]);

	  return A;
  }

  static Cvec3 lerp(Cvec3 c0, Cvec3 c1, double a) {
	  return c0 * (1.0 - a) + c1 * a;
  }

  static Quat slerp(Quat q0, Quat q1, double a) {
	  if (q0 == q1) return q0;
	  return cn(q1 * inv(q0)).power(a) * q0;
  }
  static Quat cn(Quat q) {
	  if (q[0] < 0) {
		  return Quat(-q[0], -q[1], -q[2], -q[3]);
	  }
	  else return q;
  }
};

inline RigTForm inv(const RigTForm& tform) {
  // TODO
	Quat invr = inv(tform.getRotation());
	return RigTForm(Cvec3(invr * Cvec4(tform.getTranslation(), 0)) * -1, invr);

}

inline RigTForm transFact(const RigTForm& tform) {
  return RigTForm(tform.getTranslation());
}

inline RigTForm linFact(const RigTForm& tform) {
  return RigTForm(tform.getRotation());
}

inline Matrix4 rigTFormToMatrix(const RigTForm& tform) {
  // TODO
	Matrix4 T = Matrix4::makeTranslation(tform.getTranslation());
	Matrix4 R = quatToMatrix(tform.getRotation());
  return T * R;
}

#endif
