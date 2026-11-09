/*
 *  masker.h
 *
 *  Copyright (C) 2019 Diamond Light Source
 *
 *  Author: James Parkhurst
 *
 *  This code is distributed under the GPLv3 license, a copy of 
 *  which is included in the root directory of this package.
 */

#ifndef MULTEM_MASKER_H
#define MULTEM_MASKER_H

#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <complex>
#include <stdexcept>
#include <string>
#include <vector>
#include <multem/error.h>

namespace multem {

  namespace detail {
   
    /**
     * Subtract origin rotate and then add origin
     */
    inline
    void rotate2d_around_origin(
        double ox, 
        double oy, 
        double theta, 
        double &x, 
        double &y) {
      double xt = std::cos(theta) * (x-ox) - std::sin(theta) * (y-oy) + ox;
      double yt = std::sin(theta) * (x-ox) + std::cos(theta) * (y-oy) + oy;
      x = xt;
      y = yt;
    }

    /**
     * Compute the dot product between two vectors
     */
    template <typename Vector>
    double dot(Vector a, Vector b) {
      return a[0]*b[0] + a[1]*b[1] + a[2]*b[2];
    }

    /**
     * Compute the norm squared of a vector
     */
    template <typename Vector>
    double norm_sq(Vector a) {
      return dot(a, a);
    }

    /**
     * Compute the norm of a vector
     */
    template <typename Vector>
    double norm(Vector a) {
      return std::sqrt(detail::norm_sq(a));
    }

    /**
     * Compute the cross product of a vector
     */
    template <typename Vector>
    Vector cross(Vector a, Vector b) {
      return Vector({
        a[1]*b[2] - a[2]*b[1],
        a[2]*b[0] - a[0]*b[2],
        a[0]*b[1] - a[1]*b[0]
      });
    }

    /**
     * Get the rotation between two vectors
     */
    template <typename Vector>
    Vector get_rotation_between_vectors(
        Vector a,
        Vector b) {
      
      // Compute length of vectors and normalize
      double la = detail::norm(a);
      double lb = detail::norm(b);
      MULTEM_ASSERT(la > 0 && lb > 0);
      a[0] /= la; a[1] /= la; a[2] /= la;
      b[0] /= lb; b[1] /= lb; b[2] /= lb;

      // Take the cross product to compute the rotation vector
      Vector r = detail::cross(b, a);

      // Compute angle between the vectors
      double theta = std::acos(detail::dot(b, a));
      if (theta < 0) {
        theta += 2*M_PI;
      }
      
      // Compute length of the rotation vector and return
      double lr = detail::norm(r);
      if (lr > 0) {
        r[0] = r[0] * theta / lr; 
        r[1] = r[1] * theta / lr; 
        r[2] = r[2] * theta / lr;
      }
      return r;
    }

    /**
     * Use rodriguez's formula to rotate a vector
     */
    template <typename Vector>
    Vector rotate(Vector k, Vector v) {
      double theta = detail::norm(k);
      if (theta > 0) {
        k[0] /= theta;
        k[1] /= theta;
        k[2] /= theta;
        double sin_theta = std::sin(theta);
        double cos_theta = std::cos(theta);
        double kdv = detail::dot(k, v);
        Vector kxv = detail::cross(k, v);
        v[0] = v[0] * cos_theta + kxv[0] * sin_theta + k[0] * kdv * (1 - cos_theta);
        v[1] = v[1] * cos_theta + kxv[1] * sin_theta + k[1] * kdv * (1 - cos_theta);
        v[2] = v[2] * cos_theta + kxv[2] * sin_theta + k[2] * kdv * (1 - cos_theta);
      }
      return v;
    }


    /**
     * Perform Hermite spline interpolation
     */
    class Spline {
    public:

      /**
       * Initialise the spline assuming uniform separation of points
       */
      template <typename Iterator>
      Spline(Iterator first, Iterator last)
      {
        std::size_t n = std::distance(first, last);
        MULTEM_ASSERT(n > 0);
        for (auto i = 0; i < n; ++i, ++first) {
          double A = *first;
          double B = *first;
          double C = *first;
          double D = *first;
          if (i > 0) {
            A = *(first - 1);
          }
          if (i < n - 1) {
            C = *(first + 1);
            D = *(first + 1);
          }
          if (i < n - 2) {
            D = *(first + 2);
          }
          compute_coefficients(A, B, C, D);
          a_.push_back(A);
          b_.push_back(B);
          c_.push_back(C);
          d_.push_back(D);
        }
      }

      /**
       * Interpolate the spline
       *
       * If t == 0 then the value will be Y(first) if t == 1.0 the value will
       * be Y(last)
       */
      template <typename T>
      double interpolate(T u) const {
        u = u * (size() - 1);
        int i = (int)std::floor(u);
        i = std::max(0, std::min((int)size()-1, i));
        MULTEM_ASSERT(i >= 0 && i < size())
        double t = u - i;
        return a_[i]*t*t*t + b_[i]*t*t + c_[i]*t + d_[i];
      }

      /**
       * The number of points in the spline 
       */
      std::size_t size() const {
        return a_.size();
      }

      /**
       * Get the parameters
       */
      std::vector<double> a() const {
        return a_;
      }

      /**
       * Get the parameters
       */
      std::vector<double> b() const {
        return b_;
      }
      
      /**
       * Get the parameters
       */
      std::vector<double> c() const {
        return c_;
      }
      
      /**
       * Get the parameters
       */
      std::vector<double> d() const {
        return d_;
      }

    protected:

      /**
       * Compute the spline coefficients
       */
      void compute_coefficients(double &A, double &B, double &C, double &D) const {
        double a = -A/2.0 + (3.0*B)/2.0 - (3.0*C)/2.0 + D/2.0;
        double b = A - (5.0*B)/2.0 + 2.0*C - D / 2.0;
        double c = -A/2.0 + C/2.0;
        double d = B;
        A = a;
        B = b;
        C = c;
        D = d;
      }

      std::vector<double> a_;
      std::vector<double> b_;
      std::vector<double> c_;
      std::vector<double> d_;
    };



  }

  /**
   * A cuboid masker
   */
  class CuboidMasker {
  public:

    using vector2 = std::array<double, 2>;
    using vector3 = std::array<double, 3>;
    
    CuboidMasker()
      : xsize_(0),
        ysize_(0),
        pixel_size_(1),
        offset_({0, 0, 0}),
        size_({0, 0, 0}),
        translation_({ 0, 0, 0 }),
        rotation_centre_({0, 0, 0}),
        rotation_vector_({0, 0, 0}) {
      update_geometry();    
    }

    /**
     * Get the x size of the mask image
     */
    std::size_t xsize() const {
      return xsize_;
    }

    /**
     * Get the y size of the mask image
     */
    std::size_t ysize() const {
      return ysize_;
    }

    /**
     * The total size
     */
    std::size_t image_size() const {
      return xsize() * ysize();
    }

    /**
     * Get the pixel size
     */
    double pixel_size() const {
      return pixel_size_;
    }

    /**
     * Get the offset
     */
    const vector3& offset() const {
      return offset_;
    }

    /**
     * Get the size
     */
    const vector3& size() const {
      return size_;
    }

    /**
     * Get the minimum x coordinate
     */
    double xmin() const {
      return pmin_[0];
    }
    
    /**
     * Get the minimum y coordinate
     */
    double ymin() const {
      return pmin_[1];
    }

    /**
     * Get the minimum z coordinate
     */
    double zmin() const {
      return pmin_[2];
    }
    
    /**
     * Get the maximum x coordinate
     */
    double xmax() const {
      return pmax_[0];
    }
    
    /**
     * Get the maximum y coordinate
     */
    double ymax() const {
      return pmax_[1];
    }

    /**
     * Get the maximum z coordinate
     */
    double zmax() const {
      return pmax_[2];
    }
    
    /**
     * Get the rotation origin
     */
    const vector3& rotation_origin() const {
      return rotation_centre_;
    }
    
    /**
     * Get the rotation angle
     */
    double rotation_angle() const {
      return detail::norm(rotation_vector_);
    }
    
    /**
     * Get the rotation vector
     */
    const vector3& rotation_vector() const {
      return rotation_vector_;
    }

    /**
     * Get the translation
     */
    const vector3& translation() const {
      return translation_;
    }
    
    /**
     * Get the points
     */
    const std::array<vector3, 8>& points() const {
      return points_;
    }

    /**
     * Set the image size
     */
    void set_image_size(std::size_t xsize, std::size_t ysize) {
      MULTEM_ASSERT(xsize > 0);
      MULTEM_ASSERT(ysize > 0);
      xsize_ = xsize;
      ysize_ = ysize;
      update_geometry();
    }
    
    /**
     * Set the pixel size
     */
    void set_pixel_size(double pixel_size) {
      MULTEM_ASSERT(pixel_size > 0);
      pixel_size_ = pixel_size;
      update_geometry();
    }
    
    /**
     * Set a cuboid mask
     */
    void set_parameters(std::array<double,3> offset, std::array<double,3> length) {
      MULTEM_ASSERT(length[0] > 0);
      MULTEM_ASSERT(length[1] > 0);
      MULTEM_ASSERT(length[2] > 0);
      offset_ = offset;
      size_ = length;
      update_geometry();
    }

    /**
     * Set a translation along the x axis
     */
    void set_translation(std::array<double, 3> translation) {
      translation_ = translation;
      update_geometry();
    }
    
    /**
     * Set a rotation about the x axis
     */
    void set_rotation(std::array<double, 3> centre, std::array<double, 3> rotvec) {
      rotation_centre_ = centre;
      rotation_vector_ = rotvec;
      update_geometry();
    }

    /**
     * Update the geometry
     */
    void update_geometry() {
      
      // The sample coordinate system can be rotated relative to the
      // microscope coordinate system by a rotation defined by the input
      // rotation vector and the centre of rotation. A point in the microscope
      // coordinate system can be found by rotating the sample about the
      // rotation centre
      //
      // point(microscope) = Rs * (point(sample) - centre) + centre + translation
      vector3 Rs = rotation_vector_;
      
      // The extreme points
      double x0 = offset_[0];
      double y0 = offset_[1];
      double z0 = offset_[2];
      double x1 = x0 + size_[0];
      double y1 = y0 + size_[1];
      double z1 = z0 + size_[2];
     
      // The x/y/z coords of the cuboid
      points_[0] = { x0, y0, z0 };
      points_[1] = { x1, y0, z0 };
      points_[2] = { x0, y1, z0 };
      points_[3] = { x0, y0, z1 };
      points_[4] = { x0, y1, z1 };
      points_[5] = { x1, y0, z1 };
      points_[6] = { x1, y1, z0 };
      points_[7] = { x1, y1, z1 };

      // Transform the points at the corners of the cuboid
      for (auto i = 0; i < points_.size(); ++i) {
        points_[i] = transform_cuboid_to_microscope(points_[i], Rs); 
      }

      // Compute the max and min
      pmin_ = { 1e10, 1e10, 1e10 };
      pmax_ = { -1e10, -1e10, -1e10 };
      for (auto i = 0; i < points_.size(); ++i) {
        pmin_[0] = std::min(pmin_[0], points_[i][0]);
        pmin_[1] = std::min(pmin_[1], points_[i][1]);
        pmin_[2] = std::min(pmin_[2], points_[i][2]);
        pmax_[0] = std::max(pmax_[0], points_[i][0]);
        pmax_[1] = std::max(pmax_[1], points_[i][1]);
        pmax_[2] = std::max(pmax_[2], points_[i][2]);
      }
    }
    
    /**
     * Compute the cuboid mask
     */
    template <typename Iterator>
    void compute_mask(double zs, double ze, Iterator iterator) const {
      MULTEM_ASSERT(ze > zs);

      // The middle z coordinate of the slice
      double zc = (zs + ze) / 2.0;
        
      // Get the points defining the cuboid
      auto p1 = points_[0];
      auto p2 = points_[1];
      auto p3 = points_[2];
      auto p4 = points_[3];

      // Only do something if the slice is within range
      if (zs < zmax() && ze > zmin()) {

        // Vector P1 -> P2
        auto u0 = p2[0] - p1[0];
        auto u1 = p2[1] - p1[1];
        auto u2 = p2[2] - p1[2];
        
        // Vector P1 -> P3
        auto v0 = p3[0] - p1[0];
        auto v1 = p3[1] - p1[1];
        auto v2 = p3[2] - p1[2];
        
        // Vector P1 -> P4
        auto w0 = p4[0] - p1[0];
        auto w1 = p4[1] - p1[1];
        auto w2 = p4[2] - p1[2];
       
        // Dot products u.p1, u.p2, v.p1, v.p3, w.p1, w.p4
        auto u_p1 = u0*p1[0] + u1*p1[1] + u2*p1[2];
        auto u_p2 = u0*p2[0] + u1*p2[1] + u2*p2[2];
        auto v_p1 = v0*p1[0] + v1*p1[1] + v2*p1[2];
        auto v_p3 = v0*p3[0] + v1*p3[1] + v2*p3[2];
        auto w_p1 = w0*p1[0] + w1*p1[1] + w2*p1[2];
        auto w_p4 = w0*p4[0] + w1*p4[1] + w2*p4[2];

        // order the tests
        if (u_p1 > u_p2) std::swap(u_p1, u_p2);
        if (v_p1 > v_p3) std::swap(v_p1, v_p3);
        if (w_p1 > w_p4) std::swap(w_p1, w_p4);

        // Compute the slice mask
        for (std::size_t i = 0; i < xsize_; ++i) {
          for (std::size_t j = 0; j < ysize_; ++j) {

            // The coordinate in microscope scape
            double x = i * pixel_size_;
            double y = j * pixel_size_;
            double z = zc;


            // Dot product u.x, v.x and w.x
            double u_x = u0*x + u1*y + u2*z;
            double v_x = v0*x + v1*y + v2*z;
            double w_x = w0*x + w1*y + w2*z;
      
            // Dot product must be between bounds
            if (((u_x >= u_p1) && (u_x < u_p2)) &&
                ((v_x >= v_p1) && (v_x < v_p3)) &&
                ((w_x >= w_p1) && (w_x < w_p4))) {
              *iterator = true;
            } else {
              *iterator = false;
            }
            ++iterator;
          }
        }
      } else {
        std::fill(iterator, iterator + image_size(), 0);
      }
    }

  protected:
    
    /**
     * Transform a coordinate from cuboid coordinate system to microscope
     * coordinate system
     */
    vector3 transform_cuboid_to_microscope(vector3 a, vector3 Rs) const {
        auto b = a;
        b[0] -= rotation_centre_[0];
        b[1] -= rotation_centre_[1];
        b[2] -= rotation_centre_[2];
        b = detail::rotate(Rs, b);
        b[0] += rotation_centre_[0] + translation_[0];
        b[1] += rotation_centre_[1] + translation_[1];
        b[2] += rotation_centre_[2] + translation_[2];
        return b;
    }

    std::size_t xsize_;
    std::size_t ysize_;
    double pixel_size_;
    vector3 offset_;
    vector3 size_;
    vector3 translation_;
    vector3 rotation_centre_;
    vector3 rotation_vector_;
    std::array<vector3, 8> points_;
    vector3 pmin_;
    vector3 pmax_;
  };


  /**
   * A sphere masker
   *
   * Spheres are rotation-invariant, so only the centre needs to be
   * transformed from sample to microscope coordinates.  The containment
   * test is a simple squared-distance check.
   */
  class SphereMasker {
  public:

    using vector3 = std::array<double, 3>;

    SphereMasker()
      : xsize_(0),
        ysize_(0),
        pixel_size_(1),
        centre_({0, 0, 0}),
        radius_(0),
        translation_({ 0, 0, 0 }),
        rotation_centre_({0, 0, 0}),
        rotation_vector_({0, 0, 0}),
        centre_m_({0, 0, 0}),
        pmin_({0, 0, 0}),
        pmax_({0, 0, 0}) {
      update_geometry();
    }

    std::size_t xsize() const { return xsize_; }
    std::size_t ysize() const { return ysize_; }
    std::size_t image_size() const { return xsize_ * ysize_; }
    double pixel_size() const { return pixel_size_; }

    double xmin() const { return pmin_[0]; }
    double ymin() const { return pmin_[1]; }
    double zmin() const { return pmin_[2]; }
    double xmax() const { return pmax_[0]; }
    double ymax() const { return pmax_[1]; }
    double zmax() const { return pmax_[2]; }

    const vector3& centre() const { return centre_; }
    const vector3& centre_microscope() const { return centre_m_; }  // after rotation and translation
    double radius() const { return radius_; }

    const vector3& rotation_origin() const { return rotation_centre_; }
    double rotation_angle() const { return detail::norm(rotation_vector_); }
    const vector3& rotation_vector() const { return rotation_vector_; }
    const vector3& translation() const { return translation_; }

    void set_image_size(std::size_t xsize, std::size_t ysize) {
      MULTEM_ASSERT(xsize > 0);
      MULTEM_ASSERT(ysize > 0);
      xsize_ = xsize;
      ysize_ = ysize;
      update_geometry();
    }

    void set_pixel_size(double pixel_size) {
      MULTEM_ASSERT(pixel_size > 0);
      pixel_size_ = pixel_size;
      update_geometry();
    }

    void set_parameters(vector3 centre, double radius) {
      MULTEM_ASSERT(radius > 0);
      centre_ = centre;
      radius_ = radius;
      update_geometry();
    }

    void set_translation(vector3 translation) {
      translation_ = translation;
      update_geometry();
    }

    void set_rotation(vector3 centre, vector3 rotvec) {
      rotation_centre_ = centre;
      rotation_vector_ = rotvec;
      update_geometry();
    }

    /**
     * Transform the centre to microscope coordinates and compute the AABB.
     *
     * Since a sphere is rotation-invariant, we only need to transform
     * the centre point.  The bounding box is centre_m ± radius on each axis.
     */
    void update_geometry() {
      vector3 Rs = rotation_vector_;

      // Transform centre: point(microscope) = Rs*(p - rot_centre) + rot_centre + translation
      centre_m_ = centre_;
      centre_m_[0] -= rotation_centre_[0];
      centre_m_[1] -= rotation_centre_[1];
      centre_m_[2] -= rotation_centre_[2];
      centre_m_ = detail::rotate(Rs, centre_m_);
      centre_m_[0] += rotation_centre_[0] + translation_[0];
      centre_m_[1] += rotation_centre_[1] + translation_[1];
      centre_m_[2] += rotation_centre_[2] + translation_[2];

      // AABB = centre ± radius
      pmin_ = { centre_m_[0] - radius_, centre_m_[1] - radius_, centre_m_[2] - radius_ };
      pmax_ = { centre_m_[0] + radius_, centre_m_[1] + radius_, centre_m_[2] + radius_ };
    }

    /**
     * Compute the sphere mask for a slice [zs, ze).
     *
     * At the slice midpoint zc, for each pixel (i,j) test whether
     * the squared distance to the transformed centre is <= radius^2.
     */
    template <typename Iterator>
    void compute_mask(double zs, double ze, Iterator iterator) const {
      MULTEM_ASSERT(ze > zs);

      double zc = (zs + ze) / 2.0;
      double r2 = radius_ * radius_;

      if (zs < zmax() && ze > zmin()) {
        double dz = zc - centre_m_[2];
        double dz2 = dz * dz;

        for (std::size_t i = 0; i < xsize_; ++i) {
          double x = i * pixel_size_;
          double dx = x - centre_m_[0];
          double dx2 = dx * dx;

          for (std::size_t j = 0; j < ysize_; ++j) {
            double y = j * pixel_size_;
            double dy = y - centre_m_[1];

            *iterator = (dx2 + dy * dy + dz2 <= r2);
            ++iterator;
          }
        }
      } else {
        std::fill(iterator, iterator + image_size(), 0);
      }
    }

  protected:

    std::size_t xsize_;
    std::size_t ysize_;
    double pixel_size_;
    vector3 centre_;
    double radius_;
    vector3 translation_;
    vector3 rotation_centre_;
    vector3 rotation_vector_;
    vector3 centre_m_;   // centre in microscope coordinates
    vector3 pmin_;
    vector3 pmax_;
  };


  /**
   * A cylinder masker
   */
  class CylinderMasker {
  public:

    class Parameters {
    public:
      Parameters()
        : x_a(0),
          x_b(0),
          x_c(0),
          x_d(0),
          y_a(0),
          y_b(0),
          y_c(0),
          y_d(0),
          z_a(0),
          z_b(0),
          z_c(0),
          z_d(0),
          r_a(0),
          r_b(0),
          r_c(0),
          r_d(0) {}

      Parameters(
          double x_a_,
          double x_b_,
          double x_c_,
          double x_d_,
          double y_a_,
          double y_b_,
          double y_c_,
          double y_d_,
          double z_a_,
          double z_b_,
          double z_c_,
          double z_d_,
          double r_a_,
          double r_b_,
          double r_c_,
          double r_d_)
        : x_a(x_a_),
          x_b(x_b_),
          x_c(x_c_),
          x_d(x_d_),
          y_a(y_a_),
          y_b(y_b_),
          y_c(y_c_),
          y_d(y_d_),
          z_a(z_a_),
          z_b(z_b_),
          z_c(z_c_),
          z_d(z_d_),
          r_a(r_a_),
          r_b(r_b_),
          r_c(r_c_),
          r_d(r_d_) {}

      double x_a;
      double x_b;
      double x_c;
      double x_d;
      double y_a;
      double y_b;
      double y_c;
      double y_d;
      double z_a;
      double z_b;
      double z_c;
      double z_d;
      double r_a;
      double r_b;
      double r_c;
      double r_d;
    };

    using vector3 = std::array<double, 3>;
    
    CylinderMasker()
      : xsize_(0),
        ysize_(0),
        pixel_size_(1),
        length_(0),
        origin_({0, 0, 0}),
        axis_({0, 1, 0}),
        x_offset_({0}),
        z_offset_({0}),
        radius_({1}),
        translation_({ 0, 0, 0 }),
        rotation_centre_({0, 0, 0}),
        rotation_vector_({0, 0, 0}),
        A_({0, 0, 0}),
        B_({0, 0, 0}),
        zmin_(0),
        zmax_(0) {
      update_geometry();    
    }

    /**
     * Get the x size of the mask image
     */
    std::size_t xsize() const {
      return xsize_;
    }

    /**
     * Get the y size of the mask image
     */
    std::size_t ysize() const {
      return ysize_;
    }

    /**
     * The total size
     */
    std::size_t image_size() const {
      return xsize() * ysize();
    }

    /**
     * Get the pixel size
     */
    double pixel_size() const {
      return pixel_size_;
    }

    /**
     * The start of the cylinder
     */
    vector3 A() const {
      return A_;
    }
    
    /**
     * The end of the cylinder
     */
    vector3 B() const {
      return B_;
    }

    /**
     * Get the minimum x coordinate
     */
    double xmin() const {
      return A_[0];
    }
    
    /**
     * Get the minimum y coordinate
     */
    double ymin() const {
      return A_[1];
    }

    /**
     * Get the minimum z coordinate
     */
    double zmin() const {
      return zmin_;
    }
    
    /**
     * Get the maximum x coordinate
     */
    double xmax() const {
      return B_[0];
    }
    
    /**
     * Get the maximum y coordinate
     */
    double ymax() const {
      return B_[1];
    }

    /**
     * Get the maximum z coordinate
     */
    double zmax() const {
      return zmax_;
    }
    
    /**
     * Get the rotation origin
     */
    const vector3& rotation_origin() const {
      return rotation_centre_;
    }
    
    /**
     * Get the rotation vector
     */
    const vector3& rotation_vector() const {
      return rotation_vector_;
    }
    
    /**
     * Get the rotation angle
     */
    double rotation_angle() const {
      return detail::norm(rotation_vector_);
    }

    /**
     * Get the translation
     */
    const vector3& translation() const {
      return translation_;
    }

    /**
     * Get the cylinder parameters
     */
    const std::array<CylinderMasker::Parameters, 10>& parameters() const {
      return params_;
    }

    /**
     * Set the image size
     */
    void set_image_size(std::size_t xsize, std::size_t ysize) {
      MULTEM_ASSERT(xsize > 0);
      MULTEM_ASSERT(ysize > 0);
      xsize_ = xsize;
      ysize_ = ysize;
      update_geometry();
    }
    
    /**
     * Set the pixel size
     */
    void set_pixel_size(double pixel_size) {
      MULTEM_ASSERT(pixel_size > 0);
      pixel_size_ = pixel_size;
      update_geometry();
    }
    
    /**
     * Set a cylinder parameters
     */
    void set_parameters(
        std::array<double, 3> origin, 
        std::array<double, 3> axis,
        double length,
        std::vector<double> radius,
        std::vector<double> x_offset,
        std::vector<double> z_offset) {
      MULTEM_ASSERT(length > 0);
      MULTEM_ASSERT(radius.size() == 1 || radius.size() <= 10);
      MULTEM_ASSERT(radius.size() == x_offset.size());
      MULTEM_ASSERT(radius.size() == z_offset.size());
      origin_ = origin;
      axis_ = axis;
      length_ = length;
      if (radius.size() == 1) {
        radius_.fill(radius[0]);
        x_offset_.fill(x_offset[0]);
        z_offset_.fill(z_offset[0]);
      } else if (radius.size() == 10) {
        std::copy(radius.begin(), radius.end(), radius_.begin());
        std::copy(x_offset.begin(), x_offset.end(), x_offset_.begin());
        std::copy(z_offset.begin(), z_offset.end(), z_offset_.begin());
      } else {
        auto a = (double)(radius.size() - 1) / (double)(radius_.size() - 1);
        for (auto i = 0 ; i < radius_.size(); ++i) {
          auto j = a * i;
          if (j <= 0) {
            radius_[i] = radius[0];
            x_offset_[i] = x_offset[0];
            z_offset_[i] = z_offset[0];
          } else if (j >= radius.size() - 1) {
            radius_[i] = radius[radius.size() - 1];
            x_offset_[i] = x_offset[radius.size() - 1];
            z_offset_[i] = z_offset[radius.size() - 1];
          } else {
            auto j0 = (std::size_t)std::floor(j);
            auto j1 = (std::size_t)std::ceil(j);
            auto t = j1 - j0;
            radius_[i] = (1 - t) * radius[j0] + t * radius[j1];
            x_offset_[i] = (1 - t) * x_offset[j0] + t * x_offset[j1];
            z_offset_[i] = (1 - t) * z_offset[j0] + t * z_offset[j1];
          }
        }
      }
      update_geometry();
    }

    /**
     * Set a translation along the x axis
     */
    void set_translation(std::array<double, 3> translation) {
      translation_ = translation;
      update_geometry();
    }
    
    /**
     * Set a rotation about an axis
     */
    void set_rotation(std::array<double, 3>centre, std::array<double, 3> rotvec) {
      rotation_centre_ = centre;
      rotation_vector_ = rotvec;
      update_geometry();
    }

    /**
     * Update the geometry
     */
    void update_geometry() {
      
      // The cylinder coordinate system can be both offset and rotated
      // relative to the sample coordinate system.  The offset is given by the
      // origin variable and the rotation defined by the matrix needed to
      // rotate the y axis of the sample coordinate system to the cylinder
      // coordinate system. Compute the rotation matrix needed to rotate the
      // Y-axis of the sample coordinates into Cylinder coordinates defined by
      // the axis direction of the cylinder.
      //
      // point(sample) = Rc * point(cylinder) + origin
      vector3 Rc = detail::get_rotation_between_vectors(axis_, vector3({0, 1, 0}));

      // The sample coordinate system can be rotated relative to the
      // microscope coordinate system by a rotation defined by the input
      // rotation vector and the centre of rotation. A point in the microscope
      // coordinate system can be found by rotating the sample about the
      // rotation centre
      //
      // point(microscope) = Rs * (point(sample) - centre) + centre + translation
      vector3 Rs = rotation_vector_;
      
      // Transform the points at the beginning and end of the cylinder and
      // rotate the X and Z axes of the cylinder coordinate system
      A_ = transform_cylinder_to_microscope(vector3({0, 0, 0}), Rc, Rs);
      B_ = transform_cylinder_to_microscope(vector3({0, length_, 0}), Rc, Rs);
      auto X = detail::rotate(Rs, detail::rotate(Rc, vector3({1, 0, 0})));
      auto Z = detail::rotate(Rs, detail::rotate(Rc, vector3({0, 0, 1})));

      // Now compute the cylinder offsets in microscope coordinates
      MULTEM_ASSERT(x_offset_.size() == z_offset_.size());
      MULTEM_ASSERT(x_offset_.size() == radius_.size());
      std::vector<double> Ox(x_offset_.size());
      std::vector<double> Oy(x_offset_.size());
      std::vector<double> Oz(x_offset_.size());
      for (auto i = 0; i < Ox.size(); ++i) {
        Ox[i] = x_offset_[i] * X[0] + z_offset_[i] * Z[0];
        Oy[i] = x_offset_[i] * X[1] + z_offset_[i] * Z[1];
        Oz[i] = x_offset_[i] * X[2] + z_offset_[i] * Z[2];
      }

      // Interpolate the cylinder offsets and radii between the given points
      detail::Spline interpolate_x(Ox.begin(), Ox.end());
      detail::Spline interpolate_y(Oy.begin(), Oy.end());
      detail::Spline interpolate_z(Oz.begin(), Oz.end());
      detail::Spline interpolate_r(radius_.begin(), radius_.end());

      // Compute Z min and max
      zmin_ = 1e10;
      zmax_ = -1e10;
      std::size_t num_points = 1000;
      for (auto i = 0; i < num_points; ++i) {
        double t = (double)i / (num_points-1);
        double Ox = interpolate_x.interpolate(t);
        double Oy = interpolate_y.interpolate(t);
        double Oz = interpolate_z.interpolate(t);
        double r = interpolate_r.interpolate(t);
        double x = A_[0] + t * (B_[0] - A_[0]) + Ox;
        double y = A_[1] + t * (B_[1] - A_[1]) + Oy;
        double z = A_[2] + t * (B_[2] - A_[2]) + Oz;
        zmin_ = std::min(zmin_, z - r);
        zmax_ = std::max(zmax_, z + r);
      }
      MULTEM_ASSERT(zmin_ <= zmax_);

      // Add the interpolation parameters
      MULTEM_ASSERT(interpolate_x.size() == 10);
      for (std::size_t i = 0; i < interpolate_x.size(); ++i) {
        params_[i] = CylinderMasker::Parameters(
            interpolate_x.a()[i],
            interpolate_x.b()[i],
            interpolate_x.c()[i],
            interpolate_x.d()[i],
            interpolate_y.a()[i],
            interpolate_y.b()[i],
            interpolate_y.c()[i],
            interpolate_y.d()[i],
            interpolate_z.a()[i],
            interpolate_z.b()[i],
            interpolate_z.c()[i],
            interpolate_z.d()[i],
            interpolate_r.a()[i],
            interpolate_r.b()[i],
            interpolate_r.c()[i],
            interpolate_r.d()[i]);
      }
    }
    
    /**
     * Compute the cylinder mask
     */
    template <typename Iterator>
    void compute_mask(double zs, double ze, Iterator iterator) const {
      MULTEM_ASSERT(ze > zs);

      // Extract the coordinates of the end points of the cylinder
      double Ax = A_[0];
      double Ay = A_[1];
      double Az = A_[2];
      double Bx = B_[0];
      double By = B_[1];
      double Bz = B_[2];

      for (std::size_t i = 0; i < xsize_; ++i) {
        for (std::size_t j = 0; j < ysize_; ++j) {

          // The coordinate in microscope space
          double Px = i * pixel_size_;
          double Py = j * pixel_size_;
          double Pz = (zs + ze) / 2.0;

          // Compute the position along the cylinder
          //
          // t = (P-A).(B-A) / |B-A|^2
          double t = ((Px - Ax) * (Bx - Ax) + 
                      (Py - Ay) * (By - Ay) + 
                      (Pz - Az) * (Bz - Az)) / (length_*length_); 

          // Compute the offset and radius which is a function of the distance
          // along the cylinder. Get the index and position to interpolate
          double u = t * (params_.size() - 1);
          int index = std::max(0, std::min((int)params_.size()-1, (int)std::floor(u)));
          MULTEM_ASSERT(index >= 0 && index < params_.size())
          u = u - index;
          
          // Compute the offset and radius
          const CylinderMasker::Parameters& p = params_[index];
          double u2 = u*u;
          double u3 = u2*u;
          double x_offset = p.x_a*u3 + p.x_b*u2 + p.x_c*u + p.x_d;
          double y_offset = p.y_a*u3 + p.y_b*u2 + p.y_c*u + p.y_d;
          double z_offset = p.z_a*u3 + p.z_b*u2 + p.z_c*u + p.z_d;
          double radius   = p.r_a*u3 + p.r_b*u2 + p.r_c*u + p.r_d;

          // Compute the point along the cylinder and then the distance to the
          // cylinder. The offset here is because we can incorporate local
          // deformations of the cylinder as a function of the distance along
          // the axis of the cylinder
          //
          // C = A + t * (B - A)
          // d =  | P - C |
          double Cx = Ax + t * (Bx - Ax) + x_offset;
          double Cy = Ay + t * (By - Ay) + y_offset;
          double Cz = Az + t * (Bz - Az) + z_offset;
          double d2 = (Px - Cx)*(Px - Cx) + (Py - Cy)*(Py - Cy) + (Pz - Cz)*(Pz - Cz);

          // Set if the point is within the cylinder
          *iterator++ = (t >= 0) && (t < 1) && (d2 <= (radius*radius));
        }
      }
    }

  protected:

    /**
     * Transform a coordinate from cylinder coordinate system to microscope
     * coordinate system
     */
    vector3 transform_cylinder_to_microscope(vector3 a, vector3 Rc, vector3 Rs) const {
        vector3 b = detail::rotate(Rc, a);
        b[0] += origin_[0] - rotation_centre_[0];
        b[1] += origin_[1] - rotation_centre_[1];
        b[2] += origin_[2] - rotation_centre_[2];
        b = detail::rotate(Rs, b);
        b[0] += rotation_centre_[0] + translation_[0];
        b[1] += rotation_centre_[1] + translation_[1];
        b[2] += rotation_centre_[2] + translation_[2];
        return b;
    }

    std::size_t xsize_;             // The x size of the image
    std::size_t ysize_;             // The y size of the image
    double pixel_size_;             // The pixel size

    double length_;                 // The length of the cylinder
    vector3 origin_;                // The origin of the cylinder in sample space
    vector3 axis_;                  // The axis of the cylinder in sample space
    std::array<double, 10> x_offset_;  // The x offsets as a function of y in cylinder space
    std::array<double, 10> z_offset_;  // The z offsets as a function of y in cylinder space
    std::array<double, 10> radius_;    // The radius as a function of y in cylinder space

    vector3 translation_;           // The translation
    vector3 rotation_vector_;       // The rotation vector (axis and angle)
    vector3 rotation_centre_;       // The centre of rotation

    vector3 A_;                     // The translated start point of the cylinder
    vector3 B_;                     // The translated end point of the cylinder

    double zmin_;                   // The minimum extent of the cylinder
    double zmax_;                   // The maximum extent of the cylinder

    std::array< CylinderMasker::Parameters, 10 > params_; // Interpolation parameters
  };


#ifdef __CUDACC__
#define MASKER_HD __host__ __device__
#else
#define MASKER_HD
#endif

  /**
   * A convex set: an optional quadric (p - c)^T A (p - c) <= 1 (A symmetric positive semi-definite: an
   * ellipsoid, or an elliptic cylinder when A is singular), intersected with up to MAX_PLANES half-spaces
   * n.p <= d. Microscope frame. A ray along z through (x, y) meets it in one interval [z0, z1].
   */
  struct ConvexSet {
    static const int MAX_PLANES = 32;
    bool quadric = false;
    double A[9] = {0, 0, 0, 0, 0, 0, 0, 0, 0};
    double c[3] = {0, 0, 0};
    int nplanes = 0;
    double planes[MAX_PLANES][4] = {};  // nx, ny, nz, d

    /**
     * Rounded arithmetic without fused multiply-adds, so host and device give the same bits.
     */
    MASKER_HD static double mul(double a, double b) {
#ifdef __CUDA_ARCH__
      return __dmul_rn(a, b);
#else
      return a * b;
#endif
    }

    MASKER_HD static double add(double a, double b) {
#ifdef __CUDA_ARCH__
      return __dadd_rn(a, b);
#else
      return a + b;
#endif
    }

    /**
     * The interval along z at (x, y); false if empty. Infinite ends are +-HUGE_VAL.
     */
    MASKER_HD bool interval(double x, double y, double &z0, double &z1) const {
      z0 = -HUGE_VAL;
      z1 = HUGE_VAL;
      if (quadric) {
        double dx = add(x, -c[0]);
        double dy = add(y, -c[1]);
        // q(u) = a u^2 + 2 b u + g, u = z - cz
        double a = A[8];
        double b = add(mul(A[2], dx), mul(A[5], dy));
        double g = add(add(add(mul(mul(A[0], dx), dx), mul(mul(2.0, A[1]), mul(dx, dy))), mul(mul(A[4], dy), dy)), -1.0);
        if (a > 0) {
          double disc = add(mul(b, b), -mul(a, g));
          if (!(disc >= 0)) return false;
          double s = sqrt(disc);
          z0 = add(c[2], add(-b, -s) / a);
          z1 = add(c[2], add(-b, s) / a);
        } else if (b != 0) {
          double u = -g / mul(2.0, b);           // the root of 2 b u + g = 0
          if (b > 0) z1 = add(c[2], u); else z0 = add(c[2], u);
        } else if (!(g <= 0)) {
          return false;
        }
      }
      for (int k = 0; k < nplanes; ++k) {
        double s = add(planes[k][3], -add(mul(planes[k][0], x), mul(planes[k][1], y)));
        double nz = planes[k][2];
        if (nz == 0) {
          if (!(s >= 0)) return false;
        } else if (nz > 0) {
          double t = s / nz;
          if (t < z1) z1 = t;
        } else {
          double t = s / nz;
          if (t > z0) z0 = t;
        }
      }
      return z1 > z0;
    }

    /**
     * Length of [z0, z1] inside [zs, ze)
     */
    MASKER_HD static double overlap(double z0, double z1, double zs, double ze) {
      double lo = z0 > zs ? z0 : zs;
      double hi = z1 < ze ? z1 : ze;
      return hi > lo ? add(hi, -lo) : 0.0;
    }
  };

  /**
   * A convex region with an optional convex hole: inside = in outer and not in inner. Covers ellipsoids
   * (with axes and cut planes), boxes at any rotation, slabs with rectangle or disc footprints, and
   * ellipsoidal shells (outer minus inner). The masks and the exact partial-volume fractions both come from
   * the two intervals per pixel.
   */
  struct ConvexPair {
    ConvexSet outer;
    ConvexSet inner;
    bool has_inner = false;

    MASKER_HD bool inside(double x, double y, double z) const {
      double a0, a1;
      if (!outer.interval(x, y, a0, a1) || !(z >= a0 && z < a1)) return false;
      if (has_inner) {
        double b0, b1;
        if (inner.interval(x, y, b0, b1) && z >= b0 && z < b1) return false;
      }
      return true;
    }

    MASKER_HD double fraction(double x, double y, double zs, double ze) const {
      double a0, a1;
      if (!outer.interval(x, y, a0, a1)) return 0.0;
      double f = ConvexSet::overlap(a0, a1, zs, ze);
      if (f > 0 && has_inner) {
        double b0, b1;
        if (inner.interval(x, y, b0, b1)) {
          // the part of the inner interval inside the outer one
          double lo = b0 > a0 ? b0 : a0;
          double hi = b1 < a1 ? b1 : a1;
          if (hi > lo) f = ConvexSet::add(f, -ConvexSet::overlap(lo, hi, zs, ze));
        }
      }
      return f > 0 ? f / ConvexSet::add(ze, -zs) : 0.0;
    }
  };

  /**
   * The masker for a ConvexPair given in sample coordinates; rotation and translation map it to the microscope
   * frame as for the other shapes: point(microscope) = R (point(sample) - centre) + centre + translation.
   * The bounding box is taken over the image's pixel columns (exact for what the image can see).
   */
  class ConvexMasker {
  public:

    using vector3 = std::array<double, 3>;

    ConvexMasker()
      : xsize_(0), ysize_(0), pixel_size_(1), translation_({0, 0, 0}), rotation_centre_({0, 0, 0}),
        rotation_vector_({0, 0, 0}), pmin_({0, 0, 0}), pmax_({0, 0, 0}) {}

    std::size_t xsize() const { return xsize_; }
    std::size_t ysize() const { return ysize_; }
    std::size_t image_size() const { return xsize_ * ysize_; }
    double pixel_size() const { return pixel_size_; }

    double xmin() const { extent(); return pmin_[0]; }
    double ymin() const { extent(); return pmin_[1]; }
    double zmin() const { extent(); return pmin_[2]; }
    double xmax() const { extent(); return pmax_[0]; }
    double ymax() const { extent(); return pmax_[1]; }
    double zmax() const { extent(); return pmax_[2]; }

    const ConvexPair &microscope() const { return micro_; }
    const ConvexPair &sample() const { return sample_; }

    const vector3& rotation_origin() const { return rotation_centre_; }
    double rotation_angle() const { return detail::norm(rotation_vector_); }
    const vector3& translation() const { return translation_; }

    void set_image_size(std::size_t xsize, std::size_t ysize) {
      xsize_ = xsize;
      ysize_ = ysize;
      update_geometry();
    }

    void set_pixel_size(double pixel_size) {
      MULTEM_ASSERT(pixel_size > 0);
      pixel_size_ = pixel_size;
      update_geometry();
    }

    void set_parameters(const ConvexPair &p) {
      sample_ = p;
      update_geometry();
    }

    void set_translation(vector3 translation) {
      translation_ = translation;
      update_geometry();
    }

    void set_rotation(vector3 centre, vector3 rotvec) {
      rotation_centre_ = centre;
      rotation_vector_ = rotvec;
      update_geometry();
    }

    template <typename Iterator>
    void compute_mask(double zs, double ze, Iterator iterator) const {
      MULTEM_ASSERT(ze > zs);
      double zc = (zs + ze) / 2.0;
      if (zs < zmax() && ze > zmin()) {
        for (std::size_t i = 0; i < xsize_; ++i) {
          for (std::size_t j = 0; j < ysize_; ++j) {
            *iterator = micro_.inside(ConvexSet::mul((double)i, pixel_size_), ConvexSet::mul((double)j, pixel_size_), zc);
            ++iterator;
          }
        }
      } else {
        std::fill(iterator, iterator + image_size(), 0);
      }
    }

  protected:

    static void transform(const ConvexSet &s, const double R[9], const vector3 &o, const vector3 &t, ConvexSet &m) {
      m = s;
      if (s.quadric) {
        // A_m = R A R^T; c_m = R (c - o) + o + t
        double RA[9];
        for (int r = 0; r < 3; ++r)
          for (int k = 0; k < 3; ++k)
            RA[3 * r + k] = R[3 * r + 0] * s.A[0 + k] + R[3 * r + 1] * s.A[3 + k] + R[3 * r + 2] * s.A[6 + k];
        for (int r = 0; r < 3; ++r)
          for (int k = 0; k < 3; ++k)
            m.A[3 * r + k] = RA[3 * r + 0] * R[3 * k + 0] + RA[3 * r + 1] * R[3 * k + 1] + RA[3 * r + 2] * R[3 * k + 2];
        for (int r = 0; r < 3; ++r)
          m.c[r] = R[3 * r + 0] * (s.c[0] - o[0]) + R[3 * r + 1] * (s.c[1] - o[1]) + R[3 * r + 2] * (s.c[2] - o[2]) + o[r] + t[r];
      }
      for (int k = 0; k < s.nplanes; ++k) {
        // n.p_s <= d with p_s = R^T (p_m - o - t) + o  ->  (R n).p_m <= d - n.o + (R n).(o + t)
        const double *n = s.planes[k];
        double Rn[3];
        for (int r = 0; r < 3; ++r) Rn[r] = R[3 * r + 0] * n[0] + R[3 * r + 1] * n[1] + R[3 * r + 2] * n[2];
        double d = n[3] - (n[0] * o[0] + n[1] * o[1] + n[2] * o[2])
                   + Rn[0] * (o[0] + t[0]) + Rn[1] * (o[1] + t[1]) + Rn[2] * (o[2] + t[2]);
        m.planes[k][0] = Rn[0]; m.planes[k][1] = Rn[1]; m.planes[k][2] = Rn[2]; m.planes[k][3] = d;
      }
    }

    void update_geometry() {
      double R[9];
      for (int k = 0; k < 3; ++k) {
        vector3 e = {0, 0, 0};
        e[k] = 1;
        vector3 r = detail::rotate(rotation_vector_, e);
        R[0 + k] = r[0]; R[3 + k] = r[1]; R[6 + k] = r[2];      // column k = R e_k
      }
      transform(sample_.outer, R, rotation_centre_, translation_, micro_.outer);
      transform(sample_.inner, R, rotation_centre_, translation_, micro_.inner);
      micro_.has_inner = sample_.has_inner;
      dirty_ = true;
    }

    /**
     * The bounding box over the image's pixel columns, computed when first asked for after a change (one
     * interval per pixel); an unset masker (no quadric, no planes) is empty.
     */
    void extent() const {
      if (!dirty_) return;
      dirty_ = false;
      double lo[3] = {HUGE_VAL, HUGE_VAL, HUGE_VAL}, hi[3] = {-HUGE_VAL, -HUGE_VAL, -HUGE_VAL};
      bool set = sample_.outer.quadric || sample_.outer.nplanes > 0;
      for (std::size_t i = 0; set && i < xsize_; ++i) {
        for (std::size_t j = 0; j < ysize_; ++j) {
          double x = ConvexSet::mul((double)i, pixel_size_), y = ConvexSet::mul((double)j, pixel_size_);
          double z0, z1;
          if (!micro_.outer.interval(x, y, z0, z1)) continue;
          if (x < lo[0]) lo[0] = x;
          if (x > hi[0]) hi[0] = x;
          if (y < lo[1]) lo[1] = y;
          if (y > hi[1]) hi[1] = y;
          if (z0 < lo[2]) lo[2] = z0;
          if (z1 > hi[2]) hi[2] = z1;
        }
      }
      if (!(lo[0] <= hi[0])) {
        // nothing in view: an empty extent, so a min/max over regions ignores it and no slice is in range
        pmin_ = {HUGE_VAL, HUGE_VAL, HUGE_VAL};
        pmax_ = {-HUGE_VAL, -HUGE_VAL, -HUGE_VAL};
      } else {
        MULTEM_ASSERT(std::isfinite(lo[2]) && std::isfinite(hi[2]));   // a convex region must be bounded in z
        pmin_ = {lo[0], lo[1], lo[2]};
        pmax_ = {hi[0] + pixel_size_, hi[1] + pixel_size_, hi[2]};
      }
    }

    std::size_t xsize_;
    std::size_t ysize_;
    double pixel_size_;
    vector3 translation_;
    vector3 rotation_centre_;
    vector3 rotation_vector_;
    ConvexPair sample_;
    ConvexPair micro_;
    mutable bool dirty_ = true;
    mutable vector3 pmin_;
    mutable vector3 pmax_;
  };


  /**
   * A class to represent the GRF ice parameters.
   *
   * The m0 and v0 fields allow overriding the mean potential and variance
   * calibration constants (default: calibrated against MD water model).
   * Set to a value >= 0 to override; -1 uses the built-in water defaults.
   */
  class IceParameters {
  public:

    double m1;
    double m2;
    double s1;
    double s2;
    double a1;
    double a2;
    double density;
    double m0;  // Mean potential constant override (-1 = use default 145.59)
    double v0;  // Variance constant override (-1 = use default 10195.82)

    // Optional tabulated power spectrum P(q), q in 1/A. Empty (the default) uses the two
    // Gaussians above. When set it replaces them: P is linear between knots, held at P(q0) below
    // the first knot and 0 above the last; its scale is irrelevant (v0 sets the variance). It
    // needs the band-limited potential (MULTEM_BANDLIMITED_POTENTIAL=1). With a table, v0 is the
    // variance per molecule over the table's own range: the table's integral of P 2 pi q dq in the
    // units of the target, with nothing added for power above the last knot, which counts as 0.
    std::vector<double> spectrum_q;
    std::vector<double> spectrum_P;

    IceParameters()
      : m1(0),
        m2(1.0/2.88),
        s1(0.731),
        s2(0.081),
        a1(0.199),
        a2(0.801),
        density(0.91),
        m0(-1),
        v0(-1) {}

    bool has_spectrum_table() const {
      return !spectrum_q.empty();
    }

    void set_spectrum_table(const std::vector<double> &q, const std::vector<double> &P) {
      if (q.size() != P.size()) {
        throw std::invalid_argument("spectrum table: q and P differ in length");
      }
      if (q.size() < 2) {
        throw std::invalid_argument("spectrum table: needs at least 2 knots");
      }
      bool positive = false;
      for (std::size_t i = 0; i < q.size(); ++i) {
        if (!std::isfinite(q[i]) || !std::isfinite(P[i])) {
          throw std::invalid_argument("spectrum table: values must be finite");
        }
        if (P[i] < 0) {
          throw std::invalid_argument("spectrum table: P must be >= 0");
        }
        if (i == 0 ? !(q[0] > 0) : !(q[i] > q[i-1])) {
          throw std::invalid_argument("spectrum table: q must be > 0 and strictly increasing");
        }
        positive = positive || P[i] > 0;
      }
      if (!positive) {
        throw std::invalid_argument("spectrum table: P is zero everywhere");
      }
      spectrum_q = q;
      spectrum_P = P;
    }

    void clear_spectrum_table() {
      spectrum_q.clear();
      spectrum_P.clear();
    }
  };

  /**
   * A class to generate a dynamic mask for a Z slice
   */
  class Masker {
  public:

    /**
     * The mask shapes
     */
    enum Shape {
      Cuboid,
      Cylinder,
      Sphere,
      Convex
    };

    /**
     * Initialise
     */
    Masker()
     :  shape_(Cuboid) {}

    /**
     * Initialise
     */
    Masker(std::size_t xsize, std::size_t ysize, double pixel_size)
      : shape_(Cuboid) {
      set_image_size(xsize, ysize);
      set_pixel_size(pixel_size);
    }

    /**
     * Get the cuboid masker
     */
    const CuboidMasker& cuboid_masker() const {
      return cuboid_masker_;
    }
    
    /**
     * Get the cylinder masker
     */
    const CylinderMasker& cylinder_masker() const {
      return cylinder_masker_;
    }

    /**
     * Get the sphere masker
     */
    const SphereMasker& sphere_masker() const {
      return sphere_masker_;
    }

    /**
     * Get the convex masker
     */
    const ConvexMasker& convex_masker() const {
      return convex_masker_;
    }

    /**
     * Get the x size of the mask image
     */
    std::size_t xsize() const {
      return cuboid_masker_.xsize();
    }

    /**
     * Get the y size of the mask image
     */
    std::size_t ysize() const {
      return cuboid_masker_.ysize();
    }
    
    /**
     * Get the size of the mask image
     */
    std::size_t image_size() const {
      return cuboid_masker_.image_size();
    }

    /**
     * Get the pixel size
     */
    double pixel_size() const {
      return cuboid_masker_.pixel_size();
    }

    /**
     * Get the shape to mask
     */
    Shape shape() const {
      return shape_;
    }

    /**
     * Get the minimum x coordinate
     */
    double xmin() const {
      if (shape() == Sphere) return sphere_masker_.xmin();
      if (shape() == Convex) return convex_masker_.xmin();
      return shape() == Cuboid
        ? cuboid_masker_.xmin()
        : cylinder_masker_.xmin();
    }

    /**
     * Get the minimum y coordinate
     */
    double ymin() const {
      if (shape() == Sphere) return sphere_masker_.ymin();
      if (shape() == Convex) return convex_masker_.ymin();
      return shape() == Cuboid
        ? cuboid_masker_.ymin()
        : cylinder_masker_.ymin();
    }

    /**
     * Get the minimum z coordinate
     */
    double zmin() const {
      if (shape() == Sphere) return sphere_masker_.zmin();
      if (shape() == Convex) return convex_masker_.zmin();
      return shape() == Cuboid
        ? cuboid_masker_.zmin()
        : cylinder_masker_.zmin();
    }

    /**
     * Get the maximum x coordinate
     */
    double xmax() const {
      if (shape() == Sphere) return sphere_masker_.xmax();
      if (shape() == Convex) return convex_masker_.xmax();
      return shape() == Cuboid
        ? cuboid_masker_.xmax()
        : cylinder_masker_.xmax();
    }

    /**
     * Get the maximum y coordinate
     */
    double ymax() const {
      if (shape() == Sphere) return sphere_masker_.ymax();
      if (shape() == Convex) return convex_masker_.ymax();
      return shape() == Cuboid
        ? cuboid_masker_.ymax()
        : cylinder_masker_.ymax();
    }

    /**
     * Get the maximum z coordinate
     */
    double zmax() const {
      if (shape() == Sphere) return sphere_masker_.zmax();
      if (shape() == Convex) return convex_masker_.zmax();
      return shape() == Cuboid
        ? cuboid_masker_.zmax()
        : cylinder_masker_.zmax();
    }
    
    /**
     * Get the rotation origin
     */
    const std::array<double, 3>& rotation_origin() const {
      return cuboid_masker_.rotation_origin();
    }
    
    /**
     * Get the rotation angle
     */
    double rotation_angle() const {
      return cuboid_masker_.rotation_angle();
    }

    /**
     * Get the translation
     */
    const std::array<double, 3>& translation() const {
      return cuboid_masker_.translation();
    }

    /**
     * Get the ice parameters
     */
    const IceParameters& ice_parameters() const {
      return ice_parameters_;
    }

    /**
     * Set the image size
     */
    void set_image_size(std::size_t xsize, std::size_t ysize) {
      MULTEM_ASSERT(xsize > 0);
      MULTEM_ASSERT(ysize > 0);
      cuboid_masker_.set_image_size(xsize, ysize);
      cylinder_masker_.set_image_size(xsize, ysize);
      sphere_masker_.set_image_size(xsize, ysize);
      convex_masker_.set_image_size(xsize, ysize);
    }

    /**
     * Set the pixel size
     */
    void set_pixel_size(double pixel_size) {
      MULTEM_ASSERT(pixel_size > 0);
      cuboid_masker_.set_pixel_size(pixel_size);
      cylinder_masker_.set_pixel_size(pixel_size);
      sphere_masker_.set_pixel_size(pixel_size);
      convex_masker_.set_pixel_size(pixel_size);
    }

    /**
     * Set a cube mask
     */
    void set_cube(std::array<double,3> offset, double length) {
      MULTEM_ASSERT(length > 0);
      shape_ = Cuboid;
      cuboid_masker_.set_parameters(offset, { length, length, length });
    }
    
    /**
     * Set a cuboid mask
     */
    void set_cuboid(std::array<double,3> offset, std::array<double,3> length) {
      MULTEM_ASSERT(length[0] > 0);
      MULTEM_ASSERT(length[1] > 0);
      MULTEM_ASSERT(length[2] > 0);
      shape_ = Cuboid;
      cuboid_masker_.set_parameters(offset, length);
    }

    /**
     * Set a cylinder mask
     */
    void set_cylinder(
        std::array<double, 3> origin, 
        std::array<double, 3> axis,
        double length,
        std::vector<double> radius,
        std::vector<double> x_offset,
        std::vector<double> z_offset) {
      shape_ = Cylinder;
      cylinder_masker_.set_parameters(origin, axis, length, radius, x_offset, z_offset);
    }

    /**
     * Set a sphere mask
     */
    void set_sphere(std::array<double, 3> centre, double radius) {
      MULTEM_ASSERT(radius > 0);
      shape_ = Sphere;
      sphere_masker_.set_parameters(centre, radius);
    }

    /**
     * Set a convex region (sample coordinates): an optional quadric (p - c)^T A (p - c) <= 1 (A row-major,
     * symmetric positive semi-definite; empty = none) intersected with half-spaces n.p <= d (nx, ny, nz, d),
     * minus an optional inner convex set given the same way (has_inner). Ellipsoids with axes and cuts,
     * rotated boxes, slabs and ellipsoidal shells are all of this form.
     */
    void set_convex(
        std::vector<double> A, std::array<double, 3> centre, std::vector<std::array<double, 4>> planes,
        bool has_inner, std::vector<double> inner_A, std::array<double, 3> inner_centre,
        std::vector<std::array<double, 4>> inner_planes) {
      ConvexPair p;
      fill_set(A, centre, planes, p.outer);
      p.has_inner = has_inner;
      if (has_inner) fill_set(inner_A, inner_centre, inner_planes, p.inner);
      shape_ = Convex;
      convex_masker_.set_parameters(p);
    }

    /**
     * Set a translation along the x axis
     */
    void set_translation(std::array<double, 3> translation) {
      cuboid_masker_.set_translation(translation);
      cylinder_masker_.set_translation(translation);
      sphere_masker_.set_translation(translation);
      convex_masker_.set_translation(translation);
    }

    /**
     * Set a rotation about the x axis
     */
    void set_rotation(std::array<double, 3> centre, std::array<double, 3> rotvec) {
      cuboid_masker_.set_rotation(centre, rotvec);
      cylinder_masker_.set_rotation(centre, rotvec);
      sphere_masker_.set_rotation(centre, rotvec);
      convex_masker_.set_rotation(centre, rotvec);
    }
   
    /**
     * Compute the mask
     */
    template <typename Iterator>
    void compute(double zs, double ze, Iterator iterator) const {
      if (shape_ == Cuboid) {
        cuboid_masker_.compute_mask(zs, ze, iterator);
      } else if (shape_ == Cylinder) {
        cylinder_masker_.compute_mask(zs, ze, iterator);
      } else if (shape_ == Sphere) {
        sphere_masker_.compute_mask(zs, ze, iterator);
      } else if (shape_ == Convex) {
        convex_masker_.compute_mask(zs, ze, iterator);
      } else {
        MULTEM_ASSERT(false); // Should never reach here
      }
    }

    /**
     * Set the ice parameters
     */
    void set_ice_parameters(const IceParameters &ice_parameters) {
      ice_parameters_ = ice_parameters;
    }

    /**
     * Seed the ice's random field. A seed >= 0 makes the field the same on
     * every simulation that uses this masker with the same slicing; a negative
     * seed (the default, -1) draws a fresh field each time, as before.
     */
    void set_seed(long long seed) {
      seed_ = seed;
    }

    /**
     * The random-field seed (negative: unseeded)
     */
    long long seed() const {
      return seed_;
    }

    /**
     * Partial-volume slice masks: with n > 0, each pixel of a slice is weighted by the fraction of the
     * slice's depth inside the region, instead of the 0/1 test at the slice midpoint. Spheres and cuboids
     * (at any rotation) are exact and do not use n; cylinders take n sub-slice midpoint tests (error
     * <= dz/(2n) per face). 0 (the default) keeps the 0/1 test.
     */
    void set_partial_volume(int samples) {
      MULTEM_ASSERT(samples >= 0 && samples <= 4096);
      partial_volume_ = samples;
    }

    int partial_volume() const {
      return partial_volume_;
    }

  protected:

    Shape shape_;
    CuboidMasker cuboid_masker_;
    CylinderMasker cylinder_masker_;
    SphereMasker sphere_masker_;
    ConvexMasker convex_masker_;
    IceParameters ice_parameters_;

    static void fill_set(const std::vector<double> &A, const std::array<double, 3> &c,
                         const std::vector<std::array<double, 4>> &planes, ConvexSet &s) {
      MULTEM_ASSERT(A.empty() || A.size() == 9);
      MULTEM_ASSERT((int)planes.size() <= ConvexSet::MAX_PLANES);
      s.quadric = !A.empty();
      for (int k = 0; k < 9; ++k) s.A[k] = 0.0;
      if (s.quadric) {
        // symmetric to rounding (U diag U^T computed in floating point is not exactly symmetric); symmetrise
        double amax = 0;
        for (int k = 0; k < 9; ++k) amax = std::max(amax, std::abs(A[k]));
        for (int r = 0; r < 3; ++r)
          for (int k = 0; k < 3; ++k) {
            MULTEM_ASSERT(std::abs(A[3 * r + k] - A[3 * k + r]) <= 1e-12 * amax);
            s.A[3 * r + k] = 0.5 * (A[3 * r + k] + A[3 * k + r]);
          }
        MULTEM_ASSERT(s.A[0] >= 0 && s.A[4] >= 0 && s.A[8] >= 0);
      }
      for (int k = 0; k < 3; ++k) s.c[k] = c[k];
      s.nplanes = (int)planes.size();
      for (int k = 0; k < s.nplanes; ++k)
        for (int m = 0; m < 4; ++m) s.planes[k][m] = planes[k][m];
      MULTEM_ASSERT(s.quadric || s.nplanes > 0);
    }
    long long seed_ = -1;
    int partial_volume_ = 0;
  };


  /**
   * A region within a multi-region masker.
   *
   * Each region has a shape (Masker) and material (IceParameters).
   * Optionally, another region's shape can be subtracted from this one
   * (boolean AND NOT), enabling geometries like "cuboid minus cylinder"
   * for a holey foil.
   */
  struct MaskerRegion {
    Masker shape;
    IceParameters material;
    std::vector<int> subtract_regions;  // indices of regions to subtract

    MaskerRegion()
      : subtract_regions() {}
  };


  /**
   * A masker supporting multiple spatial regions, each with its own
   * material GRF parameters.
   *
   * Enables composite geometries such as:
   *   Region 0: ice (cylinder = hole shape)
   *   Region 1: foil (cuboid minus region 0 = annulus)
   *
   * All regions share the same image size, pixel size, rotation, and
   * translation (for tilt-series consistency).
   */
  class MultiRegionMasker {
  public:

    MultiRegionMasker()
      : xsize_(0), ysize_(0), pixel_size_(1) {}

    MultiRegionMasker(std::size_t xsize, std::size_t ysize, double pixel_size)
      : xsize_(xsize), ysize_(ysize), pixel_size_(pixel_size) {}

    /**
     * Add a region with its shape and material parameters.
     * @param shape The shape primitive for this region
     * @param material The GRF parameters for this region
     * @param subtract_regions Indices of regions whose shapes are subtracted
     */
    void add_region(
        const Masker &shape,
        const IceParameters &material,
        const std::vector<int> &subtract_regions = std::vector<int>()) {
      MaskerRegion region;
      region.shape = shape;
      region.material = material;
      region.subtract_regions = subtract_regions;
      regions_.push_back(region);
    }

    /**
     * Get the number of regions
     */
    std::size_t num_regions() const {
      return regions_.size();
    }

    /**
     * Get a region by index
     */
    const MaskerRegion& region(std::size_t i) const {
      MULTEM_ASSERT(i < regions_.size());
      return regions_[i];
    }

    /**
     * Get image dimensions
     */
    std::size_t xsize() const { return xsize_; }
    std::size_t ysize() const { return ysize_; }
    std::size_t image_size() const { return xsize_ * ysize_; }
    double pixel_size() const { return pixel_size_; }

    /**
     * Get the z-extent across all regions
     */
    double zmin() const {
      double z = 1e10;
      for (auto &r : regions_) {
        z = std::min(z, r.shape.zmin());
      }
      return z;
    }

    double zmax() const {
      double z = -1e10;
      for (auto &r : regions_) {
        z = std::max(z, r.shape.zmax());
      }
      return z;
    }

    /**
     * Set rotation for ALL regions (tilt-series support)
     */
    void set_rotation(
        std::array<double, 3> centre,
        std::array<double, 3> rotvec) {
      for (auto &r : regions_) {
        r.shape.set_rotation(centre, rotvec);
      }
    }

    /**
     * Set translation for ALL regions
     */
    void set_translation(std::array<double, 3> translation) {
      for (auto &r : regions_) {
        r.shape.set_translation(translation);
      }
    }

    /**
     * Seed the random field shared by all regions (drawn in region order). A
     * seed >= 0 makes it reproducible; negative (the default, -1) draws a fresh
     * field each time, as before. The regions' own Masker seeds are not used.
     */
    void set_seed(long long seed) {
      seed_ = seed;
    }

    /**
     * The random-field seed (negative: unseeded)
     */
    long long seed() const {
      return seed_;
    }

    /**
     * Partial-volume slice masks: with n > 0, each pixel of a slice is weighted by the fraction of the
     * slice's depth inside the region, instead of the 0/1 test at the slice midpoint. Spheres and cuboids
     * (at any rotation) are exact and do not use n; cylinders take n sub-slice midpoint tests (error
     * <= dz/(2n) per face). 0 (the default) keeps the 0/1 test.
     */
    void set_partial_volume(int samples) {
      MULTEM_ASSERT(samples >= 0 && samples <= 4096);
      partial_volume_ = samples;
    }

    int partial_volume() const {
      return partial_volume_;
    }

  private:
    std::size_t xsize_;
    std::size_t ysize_;
    double pixel_size_;
    std::vector<MaskerRegion> regions_;
    long long seed_ = -1;
    int partial_volume_ = 0;
  };
}

#endif
