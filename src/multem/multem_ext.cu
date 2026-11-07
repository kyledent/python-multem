/*
 *  multem_ext.cu
 *
 *  Copyright (C) 2019 Diamond Light Source
 *
 *  Author: James Parkhurst
 *
 *  This code is distributed under the GPLv3 license, a copy of 
 *  which is included in the root directory of this package.
 */

#include <sstream>
#include <iostream>
#include <stdexcept>
#include <algorithm>
#include <array>
#include <map>
#include <string>
#include <cuda_runtime.h>
#include <thrust/random.h>
#include <types.cuh>
#include <input_multislice.cuh>
#include <xtl_build.hpp>
#include <output_multislice.hpp>
#include <tem_simulation.cuh>
#include <atomic_data.hpp>
#include <bandlimited_potential.hpp>
#include <fftw3.h>
#include <multem/multem_ext.h>

namespace multem {

  namespace detail {
  
    std::string to_string(const mt::System_Configuration &self, const std::string &prefix="") {
      std::ostringstream msg;
      msg << prefix << "precision:   " << self.precision << "\n";
      msg << prefix << "device:      " << self.device << "\n";
      msg << prefix << "cpu_ncores:  " << self.cpu_ncores << "\n";
      msg << prefix << "cpu_nthread: " << self.cpu_nthread << "\n";
      msg << prefix << "gpu_device:  " << self.gpu_device << "\n";
      msg << prefix << "gpu_nstream: " << self.gpu_nstream << "\n";
      return msg.str();
    }
    
    template <typename T>
    std::string to_string(const mt::Lens<T> &self, const std::string &prefix="") {
      std::ostringstream msg;
      msg << prefix << "m: " << self.m << "\n";
      msg << prefix << "c_10: " << self.c_10 << "\n";
      msg << prefix << "c_12: " << self.c_12 << "\n";
      msg << prefix << "phi_12: " << self.phi_12 << "\n";
      msg << prefix << "c_21: " << self.c_21 << "\n";
      msg << prefix << "phi_21: " << self.phi_21 << "\n";
      msg << prefix << "c_23: " << self.c_23 << "\n";
      msg << prefix << "phi_23: " << self.phi_23 << "\n";
      msg << prefix << "c_30: " << self.c_30 << "\n";
      msg << prefix << "c_32: " << self.c_32 << "\n";
      msg << prefix << "phi_32: " << self.phi_32 << "\n";
      msg << prefix << "c_34: " << self.c_34 << "\n";
      msg << prefix << "phi_34: " << self.phi_34 << "\n";
      msg << prefix << "c_41: " << self.c_41 << "\n";
      msg << prefix << "phi_41: " << self.phi_41 << "\n";
      msg << prefix << "c_43: " << self.c_43 << "\n";
      msg << prefix << "phi_43: " << self.phi_43 << "\n";
      msg << prefix << "c_45: " << self.c_45 << "\n";
      msg << prefix << "phi_45: " << self.phi_45 << "\n";
      msg << prefix << "c_50: " << self.c_50 << "\n";
      msg << prefix << "c_52: " << self.c_52 << "\n";
      msg << prefix << "phi_52: " << self.phi_52 << "\n";
      msg << prefix << "c_54: " << self.c_54 << "\n";
      msg << prefix << "phi_54: " << self.phi_54 << "\n";
      msg << prefix << "c_56: " << self.c_56 << "\n";
      msg << prefix << "phi_56: " << self.phi_56 << "\n";
      msg << prefix << "inner_aper_ang: " << self.inner_aper_ang << "\n";
      msg << prefix << "outer_aper_ang: " << self.outer_aper_ang << "\n";
      msg << prefix << "ti_a: " << self.ti_a << "\n";
      msg << prefix << "ti_sigma: " << self.ti_sigma << "\n";
      msg << prefix << "ti_beta: " << self.ti_beta << "\n";
      msg << prefix << "ti_npts: " << self.ti_npts << "\n";
      msg << prefix << "ti_iehwgd: " << self.ti_iehwgd << "\n";
      msg << prefix << "si_a: " << self.si_a << "\n";
      msg << prefix << "si_sigma: " << self.si_sigma << "\n";
      msg << prefix << "si_beta: " << self.si_beta << "\n";
      msg << prefix << "si_rad_npts: " << self.si_rad_npts << "\n";
      msg << prefix << "si_azm_npts: " << self.si_azm_npts << "\n";
      msg << prefix << "si_iehwgd: " << self.si_iehwgd << "\n";
      msg << prefix << "si_theta_c: " << self.si_theta_c << "\n";
      msg << prefix << "zero_defocus_type: " << self.zero_defocus_type << "\n";
      msg << prefix << "zero_defocus_plane: " << self.zero_defocus_plane << "\n";
      msg << prefix << "lambda: " << self.lambda << "\n";
      return msg.str();
    }
    
    template <typename T>
    std::string to_string(const mt::Input_Multislice<T> &self, const std::string &prefix="") {
      std::ostringstream msg;
      msg << prefix << "system_conf" << "\n";
      msg << prefix << to_string(self.system_conf, " -");
      msg << prefix << "interaction_model: " << self.interaction_model << "\n";
      msg << prefix << "potential_type: " << self.potential_type << "\n";
      msg << prefix << "pn_model: " << self.pn_model << "\n";
      msg << prefix << "pn_coh_contrib: " << self.pn_coh_contrib << "\n";
      msg << prefix << "pn_single_conf: " << self.pn_single_conf << "\n";
      /* msg << prefix << "pn_dim: " << self.pn_dim << "\n"; */
      msg << prefix << "fp_dist: " << self.fp_dist << "\n";
      msg << prefix << "pn_seed: " << self.pn_seed << "\n";
      msg << prefix << "pn_nconf: " << self.pn_nconf << "\n";
      msg << prefix << "fp_iconf_0: " << self.fp_iconf_0 << "\n";
      msg << prefix << "static_B_factor: " << self.static_B_factor << "\n";
      /* msg << prefix << Helpers<mt::AtomData<T>>::tate(self.get_atoms); */
      msg << prefix << "is_crystal: " << self.is_crystal << "\n";
      msg << prefix << "spec_rot_theta: " << self.spec_rot_theta << "\n";
      /* msg << prefix << "spec_rot_u0: " << self.spec_rot_u0 << "\n"; */
      msg << prefix << "spec_rot_center_type: " << self.spec_rot_center_type << "\n";
      /* msg << prefix << "spec_rot_center_p: " << self.spec_rot_center_p << "\n"; */
      msg << prefix << "thick_type: " << self.thick_type << "\n";
      /* msg << prefix << "thick: " << self.thick << "\n"; */
      msg << prefix << "potential_slicing: " << self.potential_slicing << "\n";
      /* msg << prefix << "grid_2d: " << self.grid_2d << "\n"; */
      /* msg << prefix << "output_area: " << self.output_area << "\n"; */
      msg << prefix << "simulation_type: " << self.simulation_type << "\n";
      msg << prefix << "iw_type: " << self.iw_type << "\n";
      /* msg << prefix << "iw_psi: " << self.iw_psi << "\n"; */
      /* msg << prefix << "iw_x: " << self.iw_x << "\n"; */
      /* msg << prefix << "iw_y: " << self.iw_y << "\n"; */
      msg << prefix << "E_0: " << self.E_0 << "\n";
      msg << prefix << "lambda: " << self.lambda << "\n";
      msg << prefix << "theta: " << self.theta << "\n";
      msg << prefix << "phi: " << self.phi << "\n";
      msg << prefix << "illumination_model: " << self.illumination_model << "\n";
      msg << prefix << "temporal_spatial_incoh: " << self.temporal_spatial_incoh << "\n";
      msg << prefix << "cond_lens" << "\n";
      msg << prefix << to_string(self.cond_lens, " -");
      msg << prefix << "obj_lens" << "\n";
      msg << prefix << to_string(self.obj_lens, " -");
      /* msg << prefix << self.scanning; */
      /* msg << prefix << self.detector; */
      /* msg << prefix << self.eels_fr; */
      msg << prefix << "operation_mode: " << self.operation_mode << "\n";
      msg << prefix << "slice_storage: " << self.slice_storage << "\n";
      msg << prefix << "reverse_multislice: " << self.reverse_multislice << "\n";
      msg << prefix << "mul_sign: " << self.mul_sign << "\n";
      msg << prefix << "Vrl: " << self.Vrl << "\n";
      msg << prefix << "nR: " << self.nR << "\n";
      msg << prefix << "nrot: " << self.nrot << "\n";
      msg << prefix << "cdl_var_type: " << self.cdl_var_type << "\n";
      /* msg << prefix << "cdl_var: " << self.cdl_var << "\n"; */
      /* msg << prefix << "iscan: " << self.iscan << "\n"; */
      /* msg << prefix << "beam_x: " << self.beam_x << "\n"; */
      /* msg << prefix << "beam_y: " << self.beam_y << "\n"; */
      msg << prefix << "islice: " << self.islice << "\n";
      msg << prefix << "dp_Shift: " << self.dp_Shift << "\n";
      return msg.str();
    }

    /**
     * Template type containing enum string info
     */
    template <typename T>
    struct EnumStrings {
      static std::string name();
      static std::map<T, std::string> items();
    };

    /**
     * Convert the enum into a string
     * @param value The enum value
     * @returns The enum string
     */
    template <
      typename T,
      typename std::enable_if<std::is_enum<T>::value>::type* = nullptr>
    std::string to_string(const T &value) {
      return EnumStrings<T>::items()[value];
    }

    /**
     * Produce an error message if the case of an unknown enum string
     * @param name The enum string
     * @returns The error message
     */
    template<
      typename T,
      typename std::enable_if<std::is_enum<T>::value>::type* = nullptr>
    std::string unknown_enum_string_message(const std::string &name) {
      std::stringstream msg;
      msg << "Unknown enum string for enum "
          << "\"" << EnumStrings<T>::name() << "\"."
          << " Got " << name << ", expected one of:\n";
      for (auto it : EnumStrings<T>::items()) {
        msg << "    " << it.second << "\n";
      }
      return msg.str();
    }

    /**
     * Convert the string into an enum value
     * @param name The enum string
     * @returns The enum value
     */
    template<
      typename T,
      typename std::enable_if<std::is_enum<T>::value>::type* = nullptr>
    T from_string(std::string name) {
      for (auto it : EnumStrings<T>::items()) {
        if (it.second == name) {
          return it.first;
        }
      }
      throw multem::Error(unknown_enum_string_message<T>(name));
    }

    /**
     * Specialization for mt::eDevice 
     */
    template <>
    struct EnumStrings<mt::eDevice> {
      
      static std::string name() {
        return "eDevice";
      }

      static std::map<mt::eDevice, std::string> items() {
        return {
          { mt::e_host, "host" },
          { mt::e_device, "device" },
          { mt::e_host_device, "host_device" },
        };
      }
    };

    /**
     * Specialization for mt::ePrecision 
     */
    template<>
    struct EnumStrings<mt::ePrecision> {
      
      static std::string name() {
        return "mt::ePrecision";
      }
      
      static std::map<mt::ePrecision, std::string> items() {
        return {
          { mt::eP_float, "float" },
          { mt::eP_double, "double" }, 
        };
      }
    };

    /**
     * Specialization for mt::eElec_Spec_Int_Model 
     */
    template<>
    struct EnumStrings<mt::eElec_Spec_Int_Model> {
      
      static std::string name() {
        return "mt::eElec_Spec_Int_Model";
      }
      
      static std::map<mt::eElec_Spec_Int_Model, std::string> items() {
        return {
          { mt::eESIM_Multislice, "Multislice" },
          { mt::eESIM_Phase_Object, "Phase_Object" },
          { mt::eESIM_Weak_Phase_Object, "Weak_Phase_Object" },
        };
      }
    };

    /**
     * Specialization for mt::ePotential_Type 
     */
    template<>
    struct EnumStrings<mt::ePotential_Type> {
      
      static std::string name() {
        return "mt::ePotential_Type";
      }
      
      static std::map<mt::ePotential_Type, std::string> items() {
        return {
          { mt::ePT_Doyle_0_4, "Doyle_0_4" },
          { mt::ePT_Peng_0_4, "Peng_0_4" },
          { mt::ePT_Peng_0_12, "Peng_0_12" },
          { mt::ePT_Kirkland_0_12, "Kirkland_0_12" },
          { mt::ePT_Weickenmeier_0_12, "Weickenmeier_0_12" },
          { mt::ePT_Lobato_0_12, "Lobato_0_12" },
          { mt::ePT_none, "none" },
        };
      }
    };

    /**
     * Specialization for mt::ePhonon_Model 
     */
    template<>
    struct EnumStrings<mt::ePhonon_Model> {
      
      static std::string name() {
        return "mt::ePhonon_Model";
      }
      
      static std::map<mt::ePhonon_Model, std::string> items() {
        return {
          { mt::ePM_Still_Atom, "Still_Atom" },
          { mt::ePM_Absorptive_Model, "Absorptive_Model" },
          { mt::ePM_Frozen_Phonon, "Frozen_Phonon" },
        };
      }
    };

    /**
     * Specialization for mt::eRot_Point_Type 
     */
    template<>
    struct EnumStrings<mt::eRot_Point_Type> {
      
      static std::string name() {
        return "mt::eRot_Point_Type";
      }
      
      static std::map<mt::eRot_Point_Type, std::string> items() {
        return {
          { mt::eRPT_geometric_center, "geometric_center" },
          { mt::eRPT_User_Define, "User_Define" },
        };
      }
    };

    /**
     * Specialization for mt::eThick_Type 
     */
    template<>
    struct EnumStrings<mt::eThick_Type> {
      
      static std::string name() {
        return "mt::eThick_Type";
      }
      
      static std::map<mt::eThick_Type, std::string> items() {
        return {
          { mt::eTT_Whole_Spec, "Whole_Spec" },
          { mt::eTT_Through_Thick, "Through_Thick" },
          { mt::eTT_Through_Slices, "Through_Slices" },
        };
      }
    };

    /**
     * Specialization for mt::ePotential_Slicing 
     */
    template<>
    struct EnumStrings<mt::ePotential_Slicing> {
      
      static std::string name() {
        return "mt::ePotential_Slicing";
      }
      
      static std::map<mt::ePotential_Slicing, std::string> items() {
        return {
          { mt::ePS_Planes, "Planes" },
          { mt::ePS_dz_Proj, "dz_Proj" },
          { mt::ePS_dz_Sub, "dz_Sub" },
          { mt::ePS_Auto, "Auto" },
        };
      }
    };

    /**
     * Specialization for mt::eTEM_Sim_Type 
     */
    template<>
    struct EnumStrings<mt::eTEM_Sim_Type> {
      
      static std::string name() {
        return "mt::eTEM_Sim_Type";
      }
      
      static std::map<mt::eTEM_Sim_Type, std::string> items() {
        return {
          { mt::eTEMST_STEM , "STEM" },
          { mt::eTEMST_ISTEM , "ISTEM" },
          { mt::eTEMST_CBED , "CBED" },
          { mt::eTEMST_CBEI , "CBEI" },
          { mt::eTEMST_ED , "ED" },
          { mt::eTEMST_HRTEM , "HRTEM" },
          { mt::eTEMST_PED , "PED" },
          { mt::eTEMST_HCTEM , "HCTEM" },
          { mt::eTEMST_EWFS , "EWFS" },
          { mt::eTEMST_EWRS , "EWRS" },
          { mt::eTEMST_EELS , "EELS" },
          { mt::eTEMST_EFTEM , "EFTEM" },
          { mt::eTEMST_IWFS , "IWFS" },
          { mt::eTEMST_IWRS , "IWRS" },
          { mt::eTEMST_PPFS , "PPFS" },
          { mt::eTEMST_PPRS , "PPRS" },
          { mt::eTEMST_TFFS , "TFFS" },
          { mt::eTEMST_TFRS , "TFRS" },
          { mt::eTEMST_PropFS , "PropFS" },
          { mt::eTEMST_PropRS , "PropRS" },
        };
      }
    };

    /**
     * Specialization for mt::eIncident_Wave_Type 
     */
    template<>
    struct EnumStrings<mt::eIncident_Wave_Type> {
      
      static std::string name() {
        return "mt::eIncident_Wave_Type";
      }
      
      static std::map<mt::eIncident_Wave_Type, std::string> items() {
        return {
          { mt::eIWT_Plane_Wave, "Plane_Wave" },
          { mt::eIWT_Convergent_Wave, "Convergent_Wave" },
          { mt::eIWT_User_Define_Wave, "User_Define_Wave" },
          { mt::eIWT_Auto, "Auto" },
        };
      }
    };

    /**
     * Specialization for mt::eIllumination_Model 
     */
    template<>
    struct EnumStrings<mt::eIllumination_Model> {
      
      static std::string name() {
        return "mt::eIllumination_Model";
      }
      
      static std::map<mt::eIllumination_Model, std::string> items() {
        return {
          { mt::eIM_Coherent, "Coherent" },
          { mt::eIM_Partial_Coherent, "Partial_Coherent" },
          { mt::eIM_Trans_Cross_Coef, "Trans_Cross_Coef" },
          { mt::eIM_Full_Integration, "Full_Integration" },
          { mt::eIM_none, "none" },
          { mt::eIM_Partial_Coherent_Higher_Order, "Partial_Coherent_Higher_Order" },
        };
      }
    };
      
    /**
     * Specialization for mt::eOperation_Mode 
     */
    template<>
    struct EnumStrings<mt::eOperation_Mode> {
      
      static std::string name() {
        return "mt::eOperation_Mode";
      }
      
      static std::map<mt::eOperation_Mode, std::string> items() {
        return {
          { mt::eOM_Normal, "Normal" },
          { mt::eOM_Advanced, "Advanced" },
        };
      }
    };

    /**
     * Specialization for mt::eLens_Var_Type 
     */
    template<>
    struct EnumStrings<mt::eLens_Var_Type> {
      
      static std::string name() {
        return "mt::eLens_Var_Type";
      }
      
      static std::map<mt::eLens_Var_Type, std::string> items() {
        return {
          { mt::eLVT_off, "off" },
          { mt::eLVT_m, "m" },
          { mt::eLVT_f, "f" },
          { mt::eLVT_Cs3, "Cs3" },
          { mt::eLVT_Cs5, "Cs5" },
          { mt::eLVT_mfa2, "mfa2" },
          { mt::eLVT_afa2, "afa2" },
          { mt::eLVT_mfa3, "mfa3" },
          { mt::eLVT_afa3, "afa3" },
          { mt::eLVT_inner_aper_ang, "inner_aper_ang" },
          { mt::eLVT_outer_aper_ang, "outer_aper_ang" },
        };
      }
    };

    /**
     * Specialization for mt::eTemporal_Spatial_Incoh 
     */
    template<>
    struct EnumStrings<mt::eTemporal_Spatial_Incoh> {
      
      static std::string name() {
        return "mt::eTemporal_Spatial_Incoh";
      }
      
      static std::map<mt::eTemporal_Spatial_Incoh, std::string> items() {
        return {
          { mt::eTSI_Temporal_Spatial, "Temporal_Spatial" },
          { mt::eTSI_Temporal, "Temporal" },
          { mt::eTSI_Spatial, "Spatial" },
          { mt::eTSI_none, "none" },
        };
      }
    };

    /**
     * Specialization for mt::eZero_Defocus_Type 
     */
    template<>
    struct EnumStrings<mt::eZero_Defocus_Type> {
      
      static std::string name() {
        return "mt::eZero_Defocus_Type";
      }
      
      static std::map<mt::eZero_Defocus_Type, std::string> items() {
        return {
          { mt::eZDT_First, "First" },
          { mt::eZDT_Middle, "Middle" },
          { mt::eZDT_Last, "Last" },
          { mt::eZDT_User_Define, "User_Define" },
        };
      }
    };

    /**
     * Specialization for mt::eScanning_Type 
     */
    template<>
    struct EnumStrings<mt::eScanning_Type> {
      
      static std::string name() {
        return "mt::eScanning_Type";
      }
      
      static std::map<mt::eScanning_Type, std::string> items() {
        return {
          { mt::eST_Line, "Line" },
          { mt::eST_Area, "Area" },
        };
      }
    };

    /**
     * Specialization for mt::eDetector_Type 
     */
    template<>
    struct EnumStrings<mt::eDetector_Type> {
      
      static std::string name() {
        return "mt::eDetector_Type";
      }
      
      static std::map<mt::eDetector_Type, std::string> items() {
        return {
          { mt::eDT_Circular, "Circular" },
          { mt::eDT_Radial, "Radial" },
          { mt::eDT_Matrix, "Matrix" },
        };
      }
    };

    /**
     * Specialization for mt::eChannelling_Type 
     */
    template<>
    struct EnumStrings<mt::eChannelling_Type> {
      
      static std::string name() {
        return "mt::eChannelling_Type";
      }
      
      static std::map<mt::eChannelling_Type, std::string> items() {
        return {
          { mt::eCT_Single_Channelling, "Single_Channelling" },
          { mt::eCT_Mixed_Channelling, "Mixed_Channelling" },
          { mt::eCT_Double_Channelling, "Double_Channelling" },
        };
      }
    };


    /**
     * Compute mask with cuboid masker
     */
    template <typename T>
    struct CuboidMaskSlice {

      double u0;
      double u1;
      double u2;
      double v0;
      double v1;
      double v2;
      double w0;
      double w1;
      double w2;
      double zc;
      double pixel_size;
      double u_p1;
      double u_p2;
      double v_p1;
      double v_p3;
      double w_p1;
      double w_p4;
      std::size_t xsize;
      std::size_t ysize;

      CuboidMaskSlice(
          double * p1_,
          double * p2_,
          double * p3_,
          double * p4_,
          double zc_,
          double pixel_size_,
          std::size_t xsize_,
          std::size_t ysize_) :
          zc(zc_),
          pixel_size(pixel_size_),
          xsize(xsize_),
          ysize(ysize_) {

        // Vector P1 -> P2
        u0 = p2_[0] - p1_[0];
        u1 = p2_[1] - p1_[1];
        u2 = p2_[2] - p1_[2];
        
        // Vector P1 -> P3
        v0 = p3_[0] - p1_[0];
        v1 = p3_[1] - p1_[1];
        v2 = p3_[2] - p1_[2];
        
        // Vector P1 -> P4
        w0 = p4_[0] - p1_[0];
        w1 = p4_[1] - p1_[1];
        w2 = p4_[2] - p1_[2];
       
        // Dot products u.p1, u.p2, v.p1, v.p3, w.p1, w.p4
        u_p1 = u0*p1_[0] + u1*p1_[1] + u2*p1_[2];
        u_p2 = u0*p2_[0] + u1*p2_[1] + u2*p2_[2];
        v_p1 = v0*p1_[0] + v1*p1_[1] + v2*p1_[2];
        v_p3 = v0*p3_[0] + v1*p3_[1] + v2*p3_[2];
        w_p1 = w0*p1_[0] + w1*p1_[1] + w2*p1_[2];
        w_p4 = w0*p4_[0] + w1*p4_[1] + w2*p4_[2];

        // order the tests
        if (u_p1 > u_p2) std::swap(u_p1, u_p2);
        if (v_p1 > v_p3) std::swap(v_p1, v_p3);
        if (w_p1 > w_p4) std::swap(w_p1, w_p4);
      }

      DEVICE_CALLABLE
      T operator()(size_t index) const {
        size_t i = index / ysize;
        size_t j = index - i * ysize;

        // The coordinate in microscope scape
        double x = i * pixel_size;
        double y = j * pixel_size;
        double z = zc;
         
        // Dot product u.x, v.x and w.x
        double u_x = u0*x + u1*y + u2*z;
        double v_x = v0*x + v1*y + v2*z;
        double w_x = w0*x + w1*y + w2*z;

        // Dot product must be between bounds
        return ((u_x >= u_p1) && (u_x < u_p2)) &&
               ((v_x >= v_p1) && (v_x < v_p3)) &&
               ((w_x >= w_p1) && (w_x < w_p4));
      }
    };

    /**
     * Compute mask with cylinder masker
     */
    template <typename T>
    struct CylinderMaskSlice {

      double Ax;
      double Ay;
      double Az;
      double Bx;
      double By;
      double Bz;
      double zs;
      double ze;
      double length;
      double pixel_size;
      std::size_t xsize;
      std::size_t ysize;
      CylinderMasker::Parameters params0;
      CylinderMasker::Parameters params1;
      CylinderMasker::Parameters params2;
      CylinderMasker::Parameters params3;
      CylinderMasker::Parameters params4;
      CylinderMasker::Parameters params5;
      CylinderMasker::Parameters params6;
      CylinderMasker::Parameters params7;
      CylinderMasker::Parameters params8;
      CylinderMasker::Parameters params9;
      std::size_t num_params;

      CylinderMaskSlice(
          double Ax_,
          double Ay_,
          double Az_,
          double Bx_,
          double By_,
          double Bz_,
          double zs_,
          double ze_,
          double pixel_size_,
          std::size_t xsize_,
          std::size_t ysize_,
          const std::array<CylinderMasker::Parameters,10>& params_)
        : Ax(Ax_),
          Ay(Ay_),
          Az(Az_),
          Bx(Bx_),
          By(By_),
          Bz(Bz_),
          zs(zs_),
          ze(ze_),
          length(std::sqrt((Bx-Ax)*(Bx-Ax)+(By-Ay)*(By-Ay)+(Bz-Az)*(Bz-Az))),
          pixel_size(pixel_size_),
          xsize(xsize_),
          ysize(ysize_),
          num_params(10) {
            params0 = params_[0];
            params1 = params_[1];
            params2 = params_[2];
            params3 = params_[3];
            params4 = params_[4];
            params5 = params_[5];
            params6 = params_[6];
            params7 = params_[7];
            params8 = params_[8];
            params9 = params_[9];
      }

      DEVICE_CALLABLE
      T operator()(size_t index) const {

        size_t i = index / ysize;
        size_t j = index - i * ysize;

        // The coordinate in microscope space
        double Px = i * pixel_size;
        double Py = j * pixel_size;
        double Pz = (zs + ze) / 2.0;

        // Compute the position along the cylinder
        //
        // t = (P-A).(B-A) / |B-A|^2
        double t = ((Px - Ax) * (Bx - Ax) + 
                    (Py - Ay) * (By - Ay) + 
                    (Pz - Az) * (Bz - Az)) / (length*length); 

        // Compute the offset and radius which is a function of the distance
        // along the cylinder
        double u = t * (num_params - 1);
        int pindex = max(0, min((int)num_params-1, (int)floor(u)));
        u = u - pindex;
        
        // Compute the offset and radius
        CylinderMasker::Parameters p = (
            pindex == 0 ? params0 :
            pindex == 1 ? params1 :
            pindex == 2 ? params2 :
            pindex == 3 ? params3 :
            pindex == 4 ? params4 : 
            pindex == 5 ? params5 : 
            pindex == 6 ? params6 : 
            pindex == 7 ? params7 : 
            pindex == 8 ? params8 : 
            params9);
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
        return (t >= 0) && (t < 1) && (d2 <= (radius*radius));
      }
    };
    
    /**
     * Compute mask with cuboid masker
     */
    template <typename Iterator>
    void compute_mask(const CuboidMasker &masker, double zs, double ze, Iterator iterator) {
      // The middle z coordinate of the slice
      double zc = (zs + ze) / 2.0;
        
      // Get the points defining the cuboid
      auto p1 = masker.points()[0];
      auto p2 = masker.points()[1];
      auto p3 = masker.points()[2];
      auto p4 = masker.points()[3];

      // Only do something if the slice is within range
      if (zs < masker.zmax() && ze > masker.zmin()) {
        thrust::counting_iterator<size_t> indices(0);
        thrust::transform(
            indices,
            indices + masker.image_size(),
            iterator, 
            CuboidMaskSlice<bool>(
              &p1[0], 
              &p2[0], 
              &p3[0], 
              &p4[0], 
              zc, 
              masker.pixel_size(), 
              masker.xsize(), 
              masker.ysize()));
      } else {
        thrust::fill(iterator, iterator + masker.image_size(), 0);
      }
    }
    
    /**
     * Compute mask with cylinder masker
     */
    template <typename Iterator>
    void compute_mask(const CylinderMasker &masker, double zs, double ze, Iterator iterator) {
      MULTEM_ASSERT(ze > zs);

      // Only do something if the slice is within range
      if (zs < masker.zmax() && ze > masker.zmin()) {
        thrust::counting_iterator<size_t> indices(0);
        thrust::transform(
            indices,
            indices + masker.image_size(),
            iterator, 
            CylinderMaskSlice<bool>(
              masker.A()[0], 
              masker.A()[1], 
              masker.A()[2], 
              masker.B()[0], 
              masker.B()[1], 
              masker.B()[2], 
              zs,
              ze,
              masker.pixel_size(),
              masker.xsize(), 
              masker.ysize(),
              masker.parameters()));
      } else {
        thrust::fill(iterator, iterator + masker.image_size(), 0);
      }
    }

    /**
     * The sphere's mask for one slice, on the device. The same test as SphereMasker::compute_mask, in
     * the same order of operations, so the mask is identical; that loop writes one element at a time,
     * which on a device vector is one transfer per pixel.
     */
    template <typename T>
    struct SphereMaskSlice {
      double cx, cy, dz2, r2, pixel_size;
      std::size_t ysize;

      SphereMaskSlice(double cx_, double cy_, double dz2_, double r2_, double pixel_size_, std::size_t ysize_)
        : cx(cx_), cy(cy_), dz2(dz2_), r2(r2_), pixel_size(pixel_size_), ysize(ysize_) {}

      DEVICE_CALLABLE
      T operator()(size_t index) const {
        size_t i = index / ysize;
        size_t j = index - i * ysize;
#ifdef __CUDA_ARCH__
        // no fused multiply-adds: the host loop's rounding, so pixels on the surface agree
        double dx = __dadd_rn(__dmul_rn((double)i, pixel_size), -cx);
        double dx2 = __dmul_rn(dx, dx);
        double dy = __dadd_rn(__dmul_rn((double)j, pixel_size), -cy);
        return (__dadd_rn(__dadd_rn(dx2, __dmul_rn(dy, dy)), dz2) <= r2);
#else
        double dx = i * pixel_size - cx;
        double dx2 = dx * dx;
        double dy = j * pixel_size - cy;
        return (dx2 + dy * dy + dz2 <= r2);
#endif
      }
    };

    /**
     * Compute mask with sphere masker
     */
    template <typename Iterator>
    void compute_mask(const SphereMasker &masker, double zs, double ze, Iterator iterator) {
      MULTEM_ASSERT(ze > zs);
      if (zs < masker.zmax() && ze > masker.zmin()) {
        const auto &c = masker.centre_microscope();
        double zc = (zs + ze) / 2.0;
        double dz = zc - c[2];
        double r = masker.radius();
        thrust::counting_iterator<size_t> indices(0);
        thrust::transform(
            indices,
            indices + masker.image_size(),
            iterator,
            SphereMaskSlice<bool>(c[0], c[1], dz * dz, r * r, masker.pixel_size(), masker.ysize()));
      } else {
        thrust::fill(iterator, iterator + masker.image_size(), 0);
      }
    }

    /**
     * Compute mask
     */
    template <typename Iterator>
    void masker_compute_mask(const Masker &masker, double zs, double ze, Iterator iterator) {
      if (masker.shape() == Masker::Cuboid) {
        compute_mask(masker.cuboid_masker(), zs, ze, iterator);
      } else if (masker.shape() == Masker::Cylinder) {
        compute_mask(masker.cylinder_masker(), zs, ze, iterator);
      } else if (masker.shape() == Masker::Sphere) {
        compute_mask(masker.sphere_masker(), zs, ze, iterator);
      } else {
        MULTEM_ASSERT(false); // Unknown shape
      }
    }

    /**
     * GPU functor for boolean AND-NOT operation on masks
     */
    struct BoolAndNot {
      DEVICE_CALLABLE
      bool operator()(bool a, bool b) const {
        return a && !b;
      }
    };

    /**
     * Partial-volume masks: f accumulates the sub-slice point tests, then is scaled by 1/n.
     */
    template <typename T>
    struct AddPointTest {
      DEVICE_CALLABLE
      T operator()(T f, bool b) const { return f + (b ? T(1) : T(0)); }
    };

    template <typename T>
    struct ScaleBy {
      T s;
      ScaleBy(T s_) : s(s_) {}
      DEVICE_CALLABLE
      T operator()(T f) const { return f * s; }
    };

    template <typename T>
    struct SubtractClip {
      DEVICE_CALLABLE
      T operator()(T a, T b) const { T d = a - b; return d > T(0) ? d : T(0); }
    };

    /**
     * The fraction of [zs, ze) inside the masker's shape, per pixel, from n sub-slice midpoint tests;
     * scratch holds one 0/1 test at a time.
     */
    template <typename FracVector, typename BoolVector>
    void masker_compute_fraction(
        const Masker &masker, double zs, double ze, int n, FracVector &frac, BoolVector &scratch) {
      using T = typename FracVector::value_type;
      MULTEM_ASSERT(n > 0 && ze > zs);
      thrust::fill(frac.begin(), frac.end(), T(0));
      double h = (ze - zs) / n;
      for (int k = 0; k < n; ++k) {
        masker_compute_mask(masker, zs + k * h, zs + (k + 1) * h, scratch.begin());
        thrust::transform(frac.begin(), frac.end(), scratch.begin(), frac.begin(), AddPointTest<T>());
      }
      thrust::transform(frac.begin(), frac.end(), frac.begin(), ScaleBy<T>(T(1.0 / n)));
    }

    /**
     * A region's partial-volume fraction with its subtracted regions taken off along the ray
     * (nested children: f = max(f_self - sum f_child, 0)).
     */
    template <typename FracVector, typename BoolVector>
    void compute_region_fraction(
        const MultiRegionMasker &masker, std::size_t region_index, double zs, double ze, int n,
        FracVector &frac, FracVector &scratch_frac, BoolVector &scratch_bool) {
      using T = typename FracVector::value_type;
      const auto &region = masker.region(region_index);
      masker_compute_fraction(region.shape, zs, ze, n, frac, scratch_bool);
      for (int sub_idx : region.subtract_regions) {
        MULTEM_ASSERT(sub_idx >= 0 && sub_idx < (int)masker.num_regions());
        masker_compute_fraction(masker.region(sub_idx).shape, zs, ze, n, scratch_frac, scratch_bool);
        thrust::transform(frac.begin(), frac.end(), scratch_frac.begin(), frac.begin(), SubtractClip<T>());
      }
    }

    /**
     * Compute the mask for a single region of a MultiRegionMasker,
     * applying boolean SUBTRACT operations.
     */
    template <typename BoolVector>
    void compute_region_mask(
        const MultiRegionMasker &masker,
        std::size_t region_index,
        double zs, double ze,
        BoolVector &output,
        BoolVector &scratch) {

      const auto &region = masker.region(region_index);

      // Compute primary shape mask
      masker_compute_mask(region.shape, zs, ze, output.begin());

      // Apply boolean SUBTRACT for each subtracted region
      for (int sub_idx : region.subtract_regions) {
        MULTEM_ASSERT(sub_idx >= 0 && sub_idx < (int)masker.num_regions());
        const auto &sub_region = masker.region(sub_idx);
        masker_compute_mask(sub_region.shape, zs, ze, scratch.begin());

        // output = output AND NOT scratch
        thrust::transform(
            output.begin(),
            output.end(),
            scratch.begin(),
            output.begin(),
            BoolAndNot());
      }
    }

    /**
     * Compute the FT of the Gaussian Random Field
     */
    template <typename T>
    struct ComputeGaussianRandomFieldAmplitude {

      double a1;
      double a2;
      double m1;
      double m2;
      double s1;
      double s2;
      double xsize;
      double ysize;
      double x_pixel_size;
      double y_pixel_size;
      double dR_bl;

      /**
       * Initialise. dR_bl > 0 multiplies the amplitude by the band-limit window
       * of a grid of that spacing (MULTEM's bandlimited_potential.hpp), so that
       * the field has the spectrum band-limited atoms would give it.
       */
      ComputeGaussianRandomFieldAmplitude(
            double a1_,
            double a2_,
            double m1_,
            double m2_,
            double s1_,
            double s2_,
            size_t xsize_, 
            size_t ysize_,
            double x_pixel_size_,
            double y_pixel_size_,
            double dR_bl_ = 0)
        : a1(a1_),
          a2(a2_),
          m1(m1_),
          m2(m2_),
          s1(s1_),
          s2(s2_),
          xsize(xsize_),
          ysize(ysize_),
          x_pixel_size(x_pixel_size_),
          y_pixel_size(y_pixel_size_),
          dR_bl(dR_bl_) {
        MULTEM_ASSERT(a1 > 0);    
        MULTEM_ASSERT(a2 > 0);    
        MULTEM_ASSERT(s1 > 0);    
        MULTEM_ASSERT(s2 > 0);    
        MULTEM_ASSERT(xsize > 0);    
        MULTEM_ASSERT(ysize > 0);    
        MULTEM_ASSERT(x_pixel_size > 0);    
        MULTEM_ASSERT(y_pixel_size > 0);    
      }

      /**
       * Compute the FT of the GRF at this index
       */
      DEVICE_CALLABLE
      T operator()(size_t index) const {
        size_t i = index / ysize;
        size_t j = index - i * ysize;

        // Compute the power spectrum and phase
        double xd = (i-xsize/2.0) / (x_pixel_size * xsize);
        double yd = (j-ysize/2.0) / (y_pixel_size * ysize);
        double r = sqrt(xd*xd+yd*yd);
        double power = 
          a1 * exp(-0.5*(r-m1)*(r-m1)/(s1*s1)) +
          a2 * exp(-0.5*(r-m2)*(r-m2)/(s2*s2));
        double amplitude = sqrt(power);
        if (dR_bl > 0) {
          amplitude *= mt::bandlimit::window(r, dR_bl);
        }
        return amplitude;
      }
    };

    /**
     * Band-limited atoms and the random field that stands in for them.
     *
     * With MULTEM_BANDLIMITED_POTENTIAL=1, MULTEM tabulates each atom's
     * projected potential band-limited to the grid (bandlimited_potential.hpp
     * in MULTEM), and an atom then deposits all of it whatever the pixel size.
     * The random field that stands in for water must do the same, so:
     *
     *  - its mean needs no pixel-size correction, and its scale M0 is the
     *    integral of the projected potential of one water molecule, from the
     *    same scattering factors the atoms use;
     *
     *  - its variance is the part of its own power spectrum P(q) that the band
     *    passes, int P W^2 d^2q / int P d^2q, and its amplitude carries W too.
     *
     * The power spectrum stands for that of projected water, correlations
     * included (its second term peaks at the first diffraction ring), so the
     * fraction is computed from it rather than from independent atoms.
     */
    inline bool band_limited_potential() {
      return mt::bandlimit::enabled();
    }

    inline double band_limit_spacing(double x_pixel_size, double y_pixel_size) {
      return mt::bandlimit::band_spacing(x_pixel_size, y_pixel_size);
    }

    /**
     * The random engine for a random-field seed. thrust's default engine is
     * minstd_rand, whose state is 1 .. 2^31-2: the seed is taken modulo 2^31-1
     * and 0 becomes 1, so seeds that differ by a multiple of 2^31-1, and the
     * seeds 0 and 1, give the same field. Callers that need distinct fields
     * should pass seeds in 1 .. 2^31-2.
     */
    inline thrust::default_random_engine seeded_random_engine(long long seed) {
      const long long m = 2147483647LL;
      long long s = seed % m;
      if (s <= 0) {
        s = (s == 0) ? 1 : s + m;
      }
      return thrust::default_random_engine(static_cast<unsigned int>(s));
    }

    /**
     * Seed a potential function's random field from its masker, if seeded.
     */
    template <typename PotentialFunction, typename MaskerType>
    void seed_from_masker(PotentialFunction &potential_function, const MaskerType &masker) {
      if (masker.seed() >= 0) {
        potential_function.set_random_engine(seeded_random_engine(masker.seed()));
      }
    }

    /**
     * c_Potf (f_e,O(0) + 2 f_e,H(0)): the projected-potential integral of one
     * water molecule, V A^3, with the Lobato scattering factors Parakeet uses.
     */
    inline double water_projected_potential_integral() {
      static const double value = []() {
        mt::Atomic_Data atomic_data(mt::ePT_Lobato_0_12);
        double sum = 0;
        for (int Z : {8, 1, 1}) {
          mt::Atom_Type<double, mt::e_host> atom_type;
          atomic_data.To_atom_type_CPU(Z, mt::c_Vrl, mt::c_nR, 1.0, atom_type);
          mt::Atom_Cal<double> atom_cal;
          atom_cal.Set_Atom_Type(mt::ePT_Lobato_0_12, 0, &atom_type);
          double fe0 = 0;
          atom_cal.feg(0.0, fe0);
          sum += mt::c_Potf * fe0;
        }
        return sum;
      }();
      return value;
    }

    /**
     * int P W^2 q dq / int P q dq for the two-Gaussian power spectrum
     * P(q) = a1 G(q; m1, s1) + a2 G(q; m2, s2) and the window W of a grid of
     * spacing dR. The denominator is analytic; the numerator is integrated with
     * a Gauss-Legendre rule on each side of the window's kink at half Nyquist.
     */
    inline double band_limited_variance_fraction_uncached(
        double a1, double a2, double m1, double m2, double s1, double s2, double dR) {
      auto power = [&](double q) {
        return a1 * std::exp(-0.5*(q-m1)*(q-m1)/(s1*s1))
             + a2 * std::exp(-0.5*(q-m2)*(q-m2)/(s2*s2));
      };
      // int_0^inf q exp(-(q-m)^2 / (2 s^2)) dq
      auto moment = [](double m, double s) {
        return s*s*std::exp(-0.5*m*m/(s*s))
             + m*s*std::sqrt(0.5*M_PI)*(1 + std::erf(m/(s*std::sqrt(2.0))));
      };
      double total = a1 * moment(m1, s1) + a2 * moment(m2, s2);
      MULTEM_ASSERT(total > 0);

      double q_nyq = 0.5 / dR;
      double edges[3] = {
        0.0,
        mt::bandlimit::c_q1_frac * q_nyq,
        mt::bandlimit::c_q2_frac * q_nyq
      };
      double band = 0;
      std::vector<double> xq, wq;
      for (int i = 0; i < 2; ++i) {
        mt::bandlimit::gauss_legendre(256, edges[i], edges[i+1], xq, wq);
        for (std::size_t k = 0; k < xq.size(); ++k) {
          double W = mt::bandlimit::window(xq[k], dR);
          band += wq[k] * xq[k] * power(xq[k]) * W * W;
        }
      }
      return band / total;
    }

    /**
     * The fraction depends only on the power spectrum and the grid spacing, which
     * are the same for every slice, but the ice asks for it once per slice and each
     * evaluation rebuilds two 256-point Gauss-Legendre rules. Remember the values
     * per thread; the result is the same computation, so it is unchanged.
     */
    inline double band_limited_variance_fraction(
        double a1, double a2, double m1, double m2, double s1, double s2, double dR) {
      thread_local std::map<std::array<double, 7>, double> cache;
      const std::array<double, 7> key = {a1, a2, m1, m2, s1, s2, dR};
      auto it = cache.find(key);
      if (it != cache.end()) {
        return it->second;
      }
      if (cache.size() > 256) {
        cache.clear();
      }
      double value = band_limited_variance_fraction_uncached(a1, a2, m1, m2, s1, s2, dR);
      cache.emplace(key, value);
      return value;
    }

    /**
     * A tabulated power spectrum (IceParameters::set_spectrum_table): linear between knots, held at
     * P(q0) below the first knot, 0 above the last.
     */
    inline double spectrum_table_power(
        double q, const std::vector<double> &qs, const std::vector<double> &Ps) {
      if (q <= qs.front()) {
        return Ps.front();
      }
      if (q >= qs.back()) {
        return 0.0;
      }
      std::size_t j = std::upper_bound(qs.begin(), qs.end(), q) - qs.begin();
      double t = (q - qs[j-1]) / (qs[j] - qs[j-1]);
      return Ps[j-1] + t * (Ps[j] - Ps[j-1]);
    }

    /**
     * int P W^2 q dq / int P q dq for a tabulated spectrum and the window of a grid of spacing dR.
     * Integrated segment by segment (the table's own knots, plus the window's kinks at Nyquist/2
     * and Nyquist), with a Gauss-Legendre rule on each piece, so no kink is smoothed over.
     */
    inline double band_limited_variance_fraction_table_uncached(
        const std::vector<double> &qs, const std::vector<double> &Ps, double dR) {
      double q_nyq = 0.5 / dR;
      std::vector<double> edges;
      edges.reserve(qs.size() + 3);
      edges.push_back(0.0);
      for (double q : qs) {
        edges.push_back(q);
      }
      edges.push_back(mt::bandlimit::c_q1_frac * q_nyq);
      edges.push_back(mt::bandlimit::c_q2_frac * q_nyq);
      std::sort(edges.begin(), edges.end());
      edges.erase(std::unique(edges.begin(), edges.end()), edges.end());
      std::vector<double> xg, wg;
      mt::bandlimit::gauss_legendre(8, -1.0, 1.0, xg, wg);
      double num = 0, den = 0;
      for (std::size_t i = 0; i + 1 < edges.size(); ++i) {
        double a = edges[i], b = edges[i+1];
        if (a >= qs.back()) {
          break;
        }
        double xm = 0.5 * (a + b), xl = 0.5 * (b - a);
        for (std::size_t k = 0; k < xg.size(); ++k) {
          double q = xm + xl * xg[k];
          double f = wg[k] * xl * q * spectrum_table_power(q, qs, Ps);
          double W = mt::bandlimit::window(q, dR);
          num += f * W * W;
          den += f;
        }
      }
      MULTEM_ASSERT(den > 0);
      return num / den;
    }

    inline double band_limited_variance_fraction_table(
        const std::vector<double> &qs, const std::vector<double> &Ps, double dR) {
      // remembered per thread, keyed by the grid spacing and the table itself (compared on a hit)
      struct Entry { double dR; std::vector<double> q, P; double value; };
      thread_local std::vector<Entry> cache;
      for (const auto &e : cache) {
        if (e.dR == dR && e.q == qs && e.P == Ps) {
          return e.value;
        }
      }
      if (cache.size() > 64) {
        cache.clear();
      }
      double value = band_limited_variance_fraction_table_uncached(qs, Ps, dR);
      cache.push_back(Entry{dR, qs, Ps, value});
      return value;
    }

    /**
     * The random field's amplitude from a tabulated spectrum, sqrt(P(r)) times the band-limit
     * window, on the same index layout as ComputeGaussianRandomFieldAmplitude. Built on the host
     * and copied once per grid.
     */
    template <typename VectorType>
    void compute_table_amplitude(
        VectorType &amplitude,
        std::size_t xsize, std::size_t ysize,
        double x_pixel_size, double y_pixel_size, double dR_bl,
        const std::vector<double> &qs, const std::vector<double> &Ps) {
      using T = typename VectorType::value_type;
      std::vector<T> host(xsize * ysize);
      for (std::size_t i = 0; i < xsize; ++i) {
        for (std::size_t j = 0; j < ysize; ++j) {
          double xd = (i - xsize/2.0) / (x_pixel_size * xsize);
          double yd = (j - ysize/2.0) / (y_pixel_size * ysize);
          double r = std::sqrt(xd*xd + yd*yd);
          double amp = std::sqrt(spectrum_table_power(r, qs, Ps));
          if (dR_bl > 0) {
            amp *= mt::bandlimit::window(r, dR_bl);
          }
          host[i * ysize + j] = amp;
        }
      }
      thrust::copy(host.begin(), host.end(), amplitude.begin());
    }

    /**
     * Compute the FT of the Gaussian Random Field
     */
    template <typename Generator, typename T>
    struct ComputeGaussianRandomField {

      Generator gen;

      /**
       * Initialise
       */
      ComputeGaussianRandomField(const Generator &gen_)
        : gen(gen_) {}

      /**
       * Compute the FT of the GRF at this index
       */
      template <typename U>
      DEVICE_CALLABLE
      T operator()(size_t index, U amplitude) const {

        // Initialise the random number generator
        Generator rnd = gen;
        rnd.discard(index);

        // The uniform distribution
        thrust::uniform_real_distribution<double> uniform(0, 2*M_PI);

        // Compute the FT of the GRF
        double phase = uniform(rnd);
        return (T)(amplitude) * exp(T(0, phase)); 
      }
    };

    /**
     * A functor to mask and normalize the gaussian field
     */
    struct MaskAndNormalize {
    
      const double a;
      const double b;

      MaskAndNormalize(double a_, double b_):
        a(a_),
        b(b_) {
      }

      template <typename T, typename U>
      DEVICE_CALLABLE
      U operator()(const T m, const U x) const {
        return m*(a * x + b);
      }
    };

    /**
     * Partial-volume version: the mean scales with the fraction f of the slice inside the region, and
     * the texture's variance with f too (its molecule count), so its amplitude with sqrt(f).
     */
    struct FractionAndNormalize {
      const double a, x_mean, mu;
      FractionAndNormalize(double a_, double x_mean_, double mu_) : a(a_), x_mean(x_mean_), mu(mu_) {}
      template <typename T, typename U>
      DEVICE_CALLABLE
      U operator()(const T f, const U x) const {
        return U(f * mu + sqrt((double)f) * a * (x - x_mean));
      }
    };

    /**
     * Add the potential and random field if potential == 0
     */
    template <typename T>
    struct AddRandomFieldAndPotential {

      T a_;
      T b_;
    
      AddRandomFieldAndPotential(T a, T b)
        : a_(a), b_(b) {}

      DEVICE_CALLABLE
      T operator()(const T r, const T p) const {
        //return p + r * b_ / (a_ * sqrt(p) + b_);
        //return p+r;//(p > 0 ? p : p + r);
        return p + (p > 0 ? r * 0.3333333 : r);
      }
    };

    /**
     * Compute an approximation of the ice potential in a slice by modelling as
     * a Gamma distribution with a certain power spectrum
     */
    template <typename FloatType, mt::eDevice DeviceType>
    class IcePotentialApproximation : public mt::PotentialFunction<FloatType, DeviceType> {
    public:
			
      using T_r = FloatType;
			using T_c = complex<FloatType>;

      thrust::default_random_engine gen_;
      double a1_;
      double a2_;
      double m1_;
      double m2_;
      double s1_;
      double s2_;
      double density_;
      double m0_;
      double v0_;
      std::vector<double> spectrum_q_;
      std::vector<double> spectrum_P_;
      double x_pixel_size_;
      double y_pixel_size_;
      mt::Grid_2d<FloatType> grid_2d_;
      mt::Vector<T_r, DeviceType> amplitude_;
      mt::Vector<T_c, DeviceType> fft_data_;
      std::size_t fft_data_counter_;
      mt::Vector<T_r, DeviceType> random_field_;
      mt::Vector<bool, DeviceType> mask_;
      mt::Vector<T_r, DeviceType> fraction_;
      mt::FFT<T_r, DeviceType> *fft_2d_;
      Masker masker_;
      
      /**
       * Initialise
       */
      IcePotentialApproximation()
        : gen_(std::random_device()()),
          m1_(0),
          m2_(1.0/2.88),
          s1_(0.731),
          s2_(0.081),
          a1_(0.199),
          a2_(0.801),
          density_(0.91),
          m0_(-1),
          v0_(-1),
          x_pixel_size_(1),
          y_pixel_size_(1),
          fft_data_counter_(0),
          fft_2d_(NULL) {}

      /**
       * Set the random engine
       */
      void set_random_engine(const thrust::default_random_engine &gen) {
        gen_ = gen;
      }

      /**
       * Get the masker
       */
      const Masker& masker() const {
        return masker_;
      }

      /**
       * Get the masker
       */
      Masker& masker() {
        return masker_;
      }

      /**
       * Set the masker
       */
      void set_masker(const Masker &masker) {
        masker_ = masker;
        m1_ = masker.ice_parameters().m1;
        m2_ = masker.ice_parameters().m2;
        s1_ = masker.ice_parameters().s1;
        s2_ = masker.ice_parameters().s2;
        a1_ = masker.ice_parameters().a1;
        a2_ = masker.ice_parameters().a2;
        density_ = masker.ice_parameters().density;
        m0_ = masker.ice_parameters().m0;
        v0_ = masker.ice_parameters().v0;
        spectrum_q_ = masker.ice_parameters().spectrum_q;
        spectrum_P_ = masker.ice_parameters().spectrum_P;
      }

      /**
       * Set the grid size
       */
      void set_grid(mt::Grid_2d<FloatType> grid_2d) {

        // If we set the grid then resize everything
        if (grid_2d.nx != grid_2d_.nx || grid_2d.ny != grid_2d_.ny) {
          grid_2d_ = grid_2d;
          amplitude_.resize(grid_2d.nx * grid_2d.ny);
          fft_data_.resize(grid_2d.nx * grid_2d.ny);
          random_field_.resize(grid_2d.nx * grid_2d.ny);
          mask_.resize(grid_2d.nx * grid_2d.ny);
          fraction_.resize(grid_2d.nx * grid_2d.ny);
          fft_data_counter_ = 0;
        }
      }

      /**
       * Set the pixel size
       */
      void set_pixel_size(double x_pixel_size, double y_pixel_size) {
        if (x_pixel_size_ != x_pixel_size || y_pixel_size_ != y_pixel_size) {
          x_pixel_size_ = x_pixel_size;
          y_pixel_size_ = y_pixel_size;
          fft_data_counter_ = 0;
        }
      }
      
      /**
       * Set the FFT instance
       */
      void set_fft_2d(mt::FFT<T_r, DeviceType> *fft_2d) {
        fft_2d_ = fft_2d;
      }

      /**
       * Compute the mask
       */
      void compute_mask(double z_0, double z_e) {

        // Check the sizes match
        MULTEM_ASSERT(masker_.xsize() == grid_2d_.nx);
        MULTEM_ASSERT(masker_.ysize() == grid_2d_.ny);
        MULTEM_ASSERT(masker_.xsize() * masker_.ysize() == mask_.size());

        // Create host vector and then compute mask and copy
        if (masker_.partial_volume() > 0) {
          masker_compute_fraction(masker_, z_0, z_e, masker_.partial_volume(), fraction_, mask_);
        } else {
          masker_compute_mask(masker_, z_0, z_e, mask_.begin());
        }
      }

      /**
       * Compute a gaussian random field
       */
      void compute_gaussian_random_field(double mu, double sigma) {
         
        // The data size
        std::size_t xsize = grid_2d_.nx;
        std::size_t ysize = grid_2d_.ny;
        std::size_t size = xsize*ysize;
       
        // Compute amplitude the first iteration
        if (fft_data_counter_ == 0 && !spectrum_q_.empty()) {
          MULTEM_ASSERT(band_limited_potential());  // a spectrum table needs the band-limited potential
          compute_table_amplitude(amplitude_, xsize, ysize, x_pixel_size_, y_pixel_size_,
              band_limit_spacing(x_pixel_size_, y_pixel_size_), spectrum_q_, spectrum_P_);
        } else if (fft_data_counter_ == 0) {
          thrust::counting_iterator<size_t> indices(0);
          thrust::transform(
              indices,
              indices + size,
              amplitude_.begin(),
              ComputeGaussianRandomFieldAmplitude<T_r>(
                a1_, 
                a2_, 
                m1_, 
                m2_, 
                s1_, 
                s2_, 
                xsize, 
                ysize,
                x_pixel_size_,
                y_pixel_size_,
                band_limited_potential()
                  ? band_limit_spacing(x_pixel_size_, y_pixel_size_)
                  : 0.0));
        }

        // We get two random fields for one calculation so we either use the
        // real or imaginary component 
        if ((fft_data_counter_ & 1) == 0) {

          // Compute the Fourier transform of the Gaussian Random Field
          thrust::counting_iterator<size_t> indices(0);
          thrust::transform(
              indices,
              indices + size,
              amplitude_.begin(),
              fft_data_.begin(),
              ComputeGaussianRandomField<thrust::default_random_engine, T_c>(gen_));
          gen_.discard(size);

          // Shift the FFT and then invert
          mt::fft2_shift(grid_2d_, fft_data_);
          fft_2d_->inverse(fft_data_);

          // Extract the real component
          mt::assign_real(fft_data_, random_field_);

        } else {
          
          // Extract the imag component
          mt::assign_imag(fft_data_, random_field_);
        }

        // Compute the mean
        MULTEM_ASSERT(random_field_.size() > 0);
        double mean = thrust::reduce(
            random_field_.begin(),
            random_field_.end(), 
            double(0), 
            mt::functor::add<double>()) / random_field_.size();

        // Compute the standard deviation
        double sdev = std::sqrt(thrust::transform_reduce(
            random_field_.begin(),
            random_field_.end(), 
            mt::functor::square_dif<double, double>(mean), 
            double(0), 
            mt::functor::add<double>()) / random_field_.size());
        
        // Normalize by the variance
        MULTEM_ASSERT(sigma > 0);
        MULTEM_ASSERT(sdev > 0);
        if (masker_.partial_volume() > 0) {
          thrust::transform(
              fraction_.begin(),
              fraction_.end(),
              random_field_.begin(),
              random_field_.begin(),
              FractionAndNormalize(sigma/sdev, mean, mu));
        } else {
          thrust::transform(
              mask_.begin(),
              mask_.end(),
              random_field_.begin(),
              random_field_.begin(),
              MaskAndNormalize(sigma/sdev, mu - mean*sigma/sdev));
        }
        
        // Toggle real/imag
        fft_data_counter_++;
      }

      /**
       * Compute the mean
       */
      double compute_mean(double density) const {
        // m0 >= 0 overrides the mean potential constant, as on the multi-region path
        if (band_limited_potential()) {
          // band-limited atoms keep their whole integral at any pixel size
          return ((m0_ >= 0) ? m0_ : water_projected_potential_integral()) * density;
        }
        //double M0 = 147.82; // Computed from MD water model
        double M0 = (m0_ >= 0) ? m0_ : 145.59; // default computed from MD water model
        double Cv = compute_mean_correction(x_pixel_size_*y_pixel_size_);
        double mean = M0 * Cv * density;
        return mean;
      };

      /**
       * Compute the sigma
       */
      double compute_sigma(double density) const {
        /* double V0 = 10784.46; // Computed by calibrating agaist MD water model */
        // double V0 = 10233.70; // Computed by calibrating agaist MD water model
        double V0 = (v0_ >= 0) ? v0_ : 10195.82; // default computed by calibrating against MD water
        if (!spectrum_q_.empty()) {
          // the legacy pixel-area correction is calibrated for the default spectrum only
          MULTEM_ASSERT(band_limited_potential());
        }
        double Cv = !spectrum_q_.empty()
          ? band_limited_variance_fraction_table(spectrum_q_, spectrum_P_,
              band_limit_spacing(x_pixel_size_, y_pixel_size_))
          : band_limited_potential()
          ? band_limited_variance_fraction(a1_, a2_, m1_, m2_, s1_, s2_,
              band_limit_spacing(x_pixel_size_, y_pixel_size_))
          : compute_variance_correction(x_pixel_size_*y_pixel_size_);
        double var = V0 * Cv * density;
        MULTEM_ASSERT(var > 0);
        return std::sqrt(var);
      }

      double compute_mean_correction(double pixel_area) const {
        std::vector<double> X = {
          0.00,
          0.01,
          0.04,
          0.09,
          0.16,
          0.25,
          0.36,
          0.49,
          0.64,
          0.81,
          1.00,
          1.21,
          1.44,
          1.69,
          1.96,
          2.25,
          2.56,
          2.89,
          3.24,
          3.61,
          4.00,
        };
        std::vector<double> Y = {
          1.0000000,
          0.9869169,
          0.9721861,
          0.9314034,
          0.8891530,
          0.8399952,
          0.7683486,
          0.6905079,
          0.6183030,
          0.5508759,
          0.4882256,
          0.4198879,
          0.3569115,
          0.3002320,
          0.2496320,
          0.2069645,
          0.1744770,
          0.1455832,
          0.1215575,
          0.1005858,
          0.0829289,
        };
        MULTEM_ASSERT(X.size() == Y.size());
        return interpolate(pixel_area, X.begin(), X.end(), Y.begin());
      }

      /**
       * Compute the pixel variance correction
       */
      double compute_variance_correction(double pixel_area) const {
        std::vector<double> X = {
          0.00,
          0.01,
          0.04,
          0.09,
          0.16,
          0.25,
          0.36,
          0.49,
          0.64,
          0.81,
          1.00,
          1.21,
          1.44,
          1.69,
          1.96,
          2.25,
          2.56,
          2.89,
          3.24,
          3.61,
          4.00,
        };
        std::vector<double> Y = {
          1.0000000,
          0.9700153,
          0.8800613,
          0.6806234,
          0.5554587,
          0.4366297,
          0.3030281,
          0.2042889,
          0.1415119,
          0.0977480,
          0.0653799,
          0.0399950,
          0.0239591,
          0.0142913,
          0.0086151,
          0.0051690,
          0.0029370,
          0.0019504,
          0.0011146,
          0.0007561,
          0.0003838,
        };
        MULTEM_ASSERT(X.size() == Y.size());
        return interpolate(pixel_area, X.begin(), X.end(), Y.begin());
      }

      /**
       * Interpolate
       */
      template <typename Iterator>
      double interpolate(double x, Iterator xfirst, Iterator xlast, Iterator yfirst) const {
        std::size_t size = xlast - xfirst;
        MULTEM_ASSERT(size >= 2);
        std::size_t index = 0;
        while ((index < size-2) && (*(xfirst+index+1) < x)) ++index;
        double x0 = *(xfirst+index);
        double y0 = *(yfirst+index);
        double x1 = *(xfirst+index+1);
        double y1 = *(yfirst+index+1);
        return (x < x0 ? y0 : (x > x1 ? y1 : (y0 + (y1 - y0)*(x - x0)/(x1 - x0))));
      }

      /**
       * Compute the number of atoms in an A^3 of water
       */
      double compute_number_density_of_water(double density) const {
        double avogadros_number = 6.02214076e+23;
        double volume = 1; // 1 A^3
        double molar_mass_of_water = 18.01528; // grams / mole
        double density_of_water = density * 1000; // g/cm3 -> kg / m^3
        double mass_of_water = (density_of_water * 1000) * (volume * std::pow(1e-10, 3)); // g
        double number_of_waters = (mass_of_water / molar_mass_of_water) * avogadros_number;
        return number_of_waters;
      }

      /**
       * Compute the Gamma random field and add to input potential
       */
      void operator()(
          double z_0,
          double z_e,
          mt::Vector<FloatType, DeviceType> &V_0) {
        
        /* std::cout << z_0 << ", " << z_e << ", " << (z_e-z_0) << std::endl; */

        // Check the sizes
        MULTEM_ASSERT(z_0 < z_e);
        MULTEM_ASSERT(grid_2d_.nx > 0 && grid_2d_.ny > 0);
        MULTEM_ASSERT(fft_2d_ != NULL);
        MULTEM_ASSERT(V_0.size() == random_field_.size());
        MULTEM_ASSERT(mask_.size() == random_field_.size());

        // The slice thickness
        double thickness = (z_e - z_0);

        // Compute the mask
        compute_mask(z_0, z_e);

        // Compute the number of particles per A^3
        double number_density = compute_number_density_of_water(density_);//0.0336; 

        // Compute the mean and sigma
        double mean = compute_mean(thickness*number_density);
        double sigma = compute_sigma(thickness*number_density);

        // Compute the Fourier transform of the Gaussian Random Field
        compute_gaussian_random_field(mean, sigma);

        // Shift the grid
        mt::fft2_shift(grid_2d_, random_field_);

        // Add the random field to the potential map
        thrust::transform(
            random_field_.begin(), 
            random_field_.end(),
            V_0.begin(),
            V_0.begin(), 
            //AddRandomFieldAndPotential<FloatType>(mean*0.66));
            AddRandomFieldAndPotential<FloatType>(1, 0.05));
      }
    };


    /**
     * Multi-material GRF potential approximation.
     *
     * Generalises IcePotentialApproximation to N spatial regions, each with
     * its own shape mask, material parameters, and independent GRF.
     * Supports boolean SUBTRACT operations between regions.
     */
    template <typename FloatType, mt::eDevice DeviceType>
    class MultiMaterialPotentialApproximation : public mt::PotentialFunction<FloatType, DeviceType> {
    public:

      using T_r = FloatType;
      using T_c = complex<FloatType>;

      /**
       * Per-region state: mask, GRF buffers, material parameters.
       */
      struct RegionState {
        double a1, a2, m1, m2, s1, s2, density, m0, v0;
        mt::Vector<bool, DeviceType> mask;
        mt::Vector<T_r, DeviceType> fraction;
        mt::Vector<T_r, DeviceType> amplitude;
        mt::Vector<T_c, DeviceType> fft_data;
        std::size_t fft_data_counter;
        mt::Vector<T_r, DeviceType> random_field;
        std::vector<double> spectrum_q;
        std::vector<double> spectrum_P;

        RegionState()
          : a1(0.199), a2(0.801), m1(0), m2(1.0/2.88),
            s1(0.731), s2(0.081), density(0.91),
            m0(-1), v0(-1), fft_data_counter(0) {}

        void set_from_ice_parameters(const IceParameters &p) {
          m1 = p.m1; m2 = p.m2;
          s1 = p.s1; s2 = p.s2;
          a1 = p.a1; a2 = p.a2;
          density = p.density;
          m0 = p.m0; v0 = p.v0;
          spectrum_q = p.spectrum_q;
          spectrum_P = p.spectrum_P;
        }

        void resize(std::size_t size) {
          mask.resize(size);
          fraction.resize(size);
          amplitude.resize(size);
          fft_data.resize(size);
          random_field.resize(size);
          fft_data_counter = 0;
        }
      };

      thrust::default_random_engine gen_;
      double x_pixel_size_;
      double y_pixel_size_;
      mt::Grid_2d<FloatType> grid_2d_;
      mt::FFT<T_r, DeviceType> *fft_2d_;
      MultiRegionMasker masker_;
      std::vector<RegionState> regions_;
      mt::Vector<bool, DeviceType> scratch_mask_;  // for boolean SUBTRACT
      mt::Vector<T_r, DeviceType> scratch_fraction_;  // for partial-volume SUBTRACT
      mt::Vector<T_r, DeviceType> region_sum_;  // all regions' fields for this slice

      MultiMaterialPotentialApproximation()
        : gen_(std::random_device()()),
          x_pixel_size_(1),
          y_pixel_size_(1),
          fft_2d_(NULL) {}

      const MultiRegionMasker& masker() const { return masker_; }
      MultiRegionMasker& masker() { return masker_; }

      void set_masker(const MultiRegionMasker &masker) {
        masker_ = masker;
        regions_.resize(masker.num_regions());
        for (std::size_t i = 0; i < masker.num_regions(); ++i) {
          regions_[i].set_from_ice_parameters(masker.region(i).material);
        }
      }

      void set_grid(mt::Grid_2d<FloatType> grid_2d) {
        if (grid_2d.nx != grid_2d_.nx || grid_2d.ny != grid_2d_.ny) {
          grid_2d_ = grid_2d;
          std::size_t size = grid_2d.nx * grid_2d.ny;
          for (auto &r : regions_) {
            r.resize(size);
          }
          scratch_mask_.resize(size);
          scratch_fraction_.resize(size);
          region_sum_.resize(size);
        }
      }

      void set_pixel_size(double x_pixel_size, double y_pixel_size) {
        if (x_pixel_size_ != x_pixel_size || y_pixel_size_ != y_pixel_size) {
          x_pixel_size_ = x_pixel_size;
          y_pixel_size_ = y_pixel_size;
          for (auto &r : regions_) {
            r.fft_data_counter = 0;
          }
        }
      }

      void set_fft_2d(mt::FFT<T_r, DeviceType> *fft_2d) {
        fft_2d_ = fft_2d;
      }

      void set_random_engine(const thrust::default_random_engine &gen) {
        gen_ = gen;
      }

      /**
       * Compute mean potential for a region, using m0 override if set.
       */
      double compute_mean(double n_particles, double m0_override) const {
        if (band_limited_potential()) {
          // band-limited atoms keep their whole integral at any pixel size
          double M0 = (m0_override >= 0) ? m0_override : water_projected_potential_integral();
          return M0 * n_particles;
        }
        double M0 = (m0_override >= 0) ? m0_override : 145.59;
        double pixel_area = x_pixel_size_ * y_pixel_size_;
        // Reuse water's correction table. It is only an approximation for another material:
        // MULTEM's own tables lose a share of each atom's potential that depends on the element.
        double Cv = compute_mean_correction(pixel_area);
        return M0 * Cv * n_particles;
      }

      double compute_sigma(double n_particles, double v0_override, const RegionState &reg) const {
        double V0 = (v0_override >= 0) ? v0_override : 10195.82;
        double pixel_area = x_pixel_size_ * y_pixel_size_;
        if (!reg.spectrum_q.empty()) {
          MULTEM_ASSERT(band_limited_potential());  // a spectrum table needs the band-limited potential
        }
        double Cv = !reg.spectrum_q.empty()
          ? band_limited_variance_fraction_table(reg.spectrum_q, reg.spectrum_P,
              band_limit_spacing(x_pixel_size_, y_pixel_size_))
          : band_limited_potential()
          ? band_limited_variance_fraction(reg.a1, reg.a2, reg.m1, reg.m2, reg.s1, reg.s2,
              band_limit_spacing(x_pixel_size_, y_pixel_size_))
          : compute_variance_correction(pixel_area);
        double var = V0 * Cv * n_particles;
        MULTEM_ASSERT(var > 0);
        return std::sqrt(var);
      }

      // Water's correction lookup tables, as in IcePotentialApproximation, for MULTEM's own
      // (not band-limited) atom tables.

      double compute_mean_correction(double pixel_area) const {
        std::vector<double> X = {
          0.00, 0.01, 0.04, 0.09, 0.16, 0.25, 0.36, 0.49, 0.64,
          0.81, 1.00, 1.21, 1.44, 1.69, 1.96, 2.25, 2.56, 2.89,
          3.24, 3.61, 4.00,
        };
        std::vector<double> Y = {
          1.0000000, 0.9869169, 0.9721861, 0.9314034, 0.8891530,
          0.8399952, 0.7683486, 0.6905079, 0.6183030, 0.5508759,
          0.4882256, 0.4198879, 0.3569115, 0.3002320, 0.2496320,
          0.2069645, 0.1744770, 0.1455832, 0.1215575, 0.1005858,
          0.0829289,
        };
        MULTEM_ASSERT(X.size() == Y.size());
        return interpolate(pixel_area, X.begin(), X.end(), Y.begin());
      }

      double compute_variance_correction(double pixel_area) const {
        std::vector<double> X = {
          0.00, 0.01, 0.04, 0.09, 0.16, 0.25, 0.36, 0.49, 0.64,
          0.81, 1.00, 1.21, 1.44, 1.69, 1.96, 2.25, 2.56, 2.89,
          3.24, 3.61, 4.00,
        };
        std::vector<double> Y = {
          1.0000000, 0.9700153, 0.8800613, 0.6806234, 0.5554587,
          0.4366297, 0.3030281, 0.2042889, 0.1415119, 0.0977480,
          0.0653799, 0.0399950, 0.0239591, 0.0142913, 0.0086151,
          0.0051690, 0.0029370, 0.0019504, 0.0011146, 0.0007561,
          0.0003838,
        };
        MULTEM_ASSERT(X.size() == Y.size());
        return interpolate(pixel_area, X.begin(), X.end(), Y.begin());
      }

      template <typename Iterator>
      double interpolate(double x, Iterator xfirst, Iterator xlast, Iterator yfirst) const {
        std::size_t size = xlast - xfirst;
        MULTEM_ASSERT(size >= 2);
        std::size_t index = 0;
        while ((index < size-2) && (*(xfirst+index+1) < x)) ++index;
        double x0 = *(xfirst+index);
        double y0 = *(yfirst+index);
        double x1 = *(xfirst+index+1);
        double y1 = *(yfirst+index+1);
        return (x < x0 ? y0 : (x > x1 ? y1 : (y0 + (y1 - y0)*(x - x0)/(x1 - x0))));
      }

      double compute_number_density_of_water(double density) const {
        double avogadros_number = 6.02214076e+23;
        double volume = 1;
        double molar_mass_of_water = 18.01528;
        double density_of_water = density * 1000;
        double mass_of_water = (density_of_water * 1000) * (volume * std::pow(1e-10, 3));
        double number_of_waters = (mass_of_water / molar_mass_of_water) * avogadros_number;
        return number_of_waters;
      }

      /**
       * Generate a GRF for a single region, masked and normalized.
       */
      void compute_grf_for_region(
          RegionState &reg,
          double mean, double sigma) {

        std::size_t xsize = grid_2d_.nx;
        std::size_t ysize = grid_2d_.ny;
        std::size_t size = xsize * ysize;

        // Compute amplitude on first call (or after grid resize)
        if (reg.fft_data_counter == 0 && !reg.spectrum_q.empty()) {
          MULTEM_ASSERT(band_limited_potential());
          compute_table_amplitude(reg.amplitude, xsize, ysize, x_pixel_size_, y_pixel_size_,
              band_limit_spacing(x_pixel_size_, y_pixel_size_), reg.spectrum_q, reg.spectrum_P);
        } else if (reg.fft_data_counter == 0) {
          thrust::counting_iterator<size_t> indices(0);
          thrust::transform(
              indices, indices + size,
              reg.amplitude.begin(),
              ComputeGaussianRandomFieldAmplitude<T_r>(
                reg.a1, reg.a2, reg.m1, reg.m2, reg.s1, reg.s2,
                xsize, ysize, x_pixel_size_, y_pixel_size_,
                band_limited_potential()
                  ? band_limit_spacing(x_pixel_size_, y_pixel_size_)
                  : 0.0));
        }

        // Generate two random fields per FFT (real + imag)
        if ((reg.fft_data_counter & 1) == 0) {
          thrust::counting_iterator<size_t> indices(0);
          thrust::transform(
              indices, indices + size,
              reg.amplitude.begin(),
              reg.fft_data.begin(),
              ComputeGaussianRandomField<thrust::default_random_engine, T_c>(gen_));
          gen_.discard(size);

          mt::fft2_shift(grid_2d_, reg.fft_data);
          fft_2d_->inverse(reg.fft_data);
          mt::assign_real(reg.fft_data, reg.random_field);
        } else {
          mt::assign_imag(reg.fft_data, reg.random_field);
        }

        // Normalize
        MULTEM_ASSERT(reg.random_field.size() > 0);
        double rf_mean = thrust::reduce(
            reg.random_field.begin(), reg.random_field.end(),
            double(0), mt::functor::add<double>()) / reg.random_field.size();

        double sdev = std::sqrt(thrust::transform_reduce(
            reg.random_field.begin(), reg.random_field.end(),
            mt::functor::square_dif<double, double>(rf_mean),
            double(0), mt::functor::add<double>()) / reg.random_field.size());

        MULTEM_ASSERT(sigma > 0);
        MULTEM_ASSERT(sdev > 0);

        // Mask and normalize
        if (masker_.partial_volume() > 0) {
          thrust::transform(
              reg.fraction.begin(), reg.fraction.end(),
              reg.random_field.begin(),
              reg.random_field.begin(),
              FractionAndNormalize(sigma/sdev, rf_mean, mean));
        } else {
          thrust::transform(
              reg.mask.begin(), reg.mask.end(),
              reg.random_field.begin(),
              reg.random_field.begin(),
              MaskAndNormalize(sigma/sdev, mean - rf_mean*sigma/sdev));
        }

        reg.fft_data_counter++;
      }

      /**
       * Per-slice callback: compute all region GRFs and add to potential.
       */
      void operator()(
          double z_0,
          double z_e,
          mt::Vector<FloatType, DeviceType> &V_0) {

        MULTEM_ASSERT(z_0 < z_e);
        MULTEM_ASSERT(grid_2d_.nx > 0 && grid_2d_.ny > 0);
        MULTEM_ASSERT(fft_2d_ != NULL);

        double thickness = z_e - z_0;

        // Sum every region's field first, then add the sum to the potential once. The upstream rule
        // (ice at a third where the potential is positive) is meant for the atoms' potential; added
        // region by region it also saw earlier regions' fields wherever two regions share a pixel,
        // which partial-volume masks make the rule at faces. With 0/1 masks no two regions share a
        // pixel, so the result is unchanged.
        MULTEM_ASSERT(region_sum_.size() == V_0.size());
        thrust::fill(region_sum_.begin(), region_sum_.end(), T_r(0));

        for (std::size_t i = 0; i < regions_.size(); ++i) {
          auto &reg = regions_[i];

          // A vacuum region (density <= 0) adds nothing. It exists so that other regions can
          // subtract its shape, which compute_region_mask reads from the masker, not from here.
          if (!(reg.density > 0)) {
            continue;
          }

          MULTEM_ASSERT(V_0.size() == reg.random_field.size());
          MULTEM_ASSERT(reg.mask.size() == reg.random_field.size());

          // Compute mask with boolean subtraction (or the partial-volume fraction)
          if (masker_.partial_volume() > 0) {
            compute_region_fraction(masker_, i, z_0, z_e, masker_.partial_volume(),
                                    reg.fraction, scratch_fraction_, scratch_mask_);
          } else {
            compute_region_mask(masker_, i, z_0, z_e, reg.mask, scratch_mask_);
          }

          // Compute GRF statistics for this material
          double number_density = compute_number_density_of_water(reg.density);
          double mean = compute_mean(thickness * number_density, reg.m0);
          double sigma = compute_sigma(thickness * number_density, reg.v0, reg);

          // Generate and mask the GRF
          compute_grf_for_region(reg, mean, sigma);

          // Shift the grid
          mt::fft2_shift(grid_2d_, reg.random_field);

          // Add to this slice's sum of region fields
          thrust::transform(
              reg.random_field.begin(),
              reg.random_field.end(),
              region_sum_.begin(),
              region_sum_.begin(),
              thrust::plus<T_r>());
        }

        // Add the regions to the potential
        thrust::transform(
            region_sum_.begin(),
            region_sum_.end(),
            V_0.begin(),
            V_0.begin(),
            AddRandomFieldAndPotential<FloatType>(1, 0.05));
      }
    };


    /**
     * Run the multislice simulation
     * @param system_conf The system configuration
     * @param input_multislice The input object
     * @param output_multislice The output object
     */
    template <typename FloatType, mt::eDevice DeviceType>
    void run_multislice_internal(
      const mt::System_Configuration &system_conf,
      mt::Input_Multislice<FloatType> &input_multislice,
      mt::Output_Multislice<FloatType> &output_multislice) {
  
      // Set the system configration    
      input_multislice.system_conf = system_conf;

      // Open a stream
      mt::Stream<DeviceType> stream(system_conf.nstream);

      // Create the FFT object
      mt::FFT<FloatType, DeviceType> fft_2d;
      fft_2d.create_plan_2d(
        input_multislice.grid_2d.ny, 
        input_multislice.grid_2d.nx, 
        system_conf.nstream);

      // Setup the multislice simulation 
      mt::Multislice<FloatType, DeviceType> multislice;
      multislice.set_input_data(&input_multislice, &stream, &fft_2d);
      /* std::cout << to_string(input_multislice) << std::endl; */

      // Set the input data
      output_multislice.set_input_data(&input_multislice);

      // Perform the multislice simulation
      multislice(output_multislice);
      stream.synchronize();

      // Get the results
      output_multislice.gather();
      output_multislice.clean_temporal();
      fft_2d.cleanup();

      // If there was an error then throw an exception
      if (DeviceType == mt::e_device) {
        auto err = cudaGetLastError();
        if (err != cudaSuccess) {
          throw multem::Error(__FILE__, __LINE__, cudaGetErrorString(err));
        }
      }
    }
    
    /**
     * Run the multislice simulation
     * @param system_conf The system configuration
     * @param input_multislice The input object
     * @param output_multislice The output object
     */
    template <typename FloatType, mt::eDevice DeviceType, typename Masker>
    void run_multislice_internal(
      const mt::System_Configuration &system_conf,
      mt::Input_Multislice<FloatType> &input_multislice,
      mt::Output_Multislice<FloatType> &output_multislice,
      const Masker &masker) {
  
      // Set the system configration    
      input_multislice.system_conf = system_conf;

      // Open a stream
      mt::Stream<DeviceType> stream(system_conf.nstream);

      // Create the FFT object
      mt::FFT<FloatType, DeviceType> fft_2d;
      fft_2d.create_plan_2d(
        input_multislice.grid_2d.ny, 
        input_multislice.grid_2d.nx, 
        system_conf.nstream);

      // Setup the multislice simulation 
      mt::Multislice<FloatType, DeviceType> multislice;
      multislice.set_input_data(&input_multislice, &stream, &fft_2d);
   
      // Compute the pixel sizes
      double x_pixel_size = input_multislice.atoms.l_x / input_multislice.grid_2d.nx;
      double y_pixel_size = input_multislice.atoms.l_y / input_multislice.grid_2d.ny;

      // Setup the ice potential approximation
      IcePotentialApproximation<FloatType, DeviceType> potential_function;
      potential_function.set_fft_2d(&fft_2d);
      potential_function.set_grid(input_multislice.grid_2d);
      potential_function.set_masker(masker);
      seed_from_masker(potential_function, masker);
      potential_function.set_pixel_size(x_pixel_size, y_pixel_size);
      multislice.set_potential_function(&potential_function);

      // Set the input data
      output_multislice.set_input_data(&input_multislice);

      // Perform the multislice simulation
      multislice(output_multislice);
      stream.synchronize();

      // Get the results
      output_multislice.gather();
      output_multislice.clean_temporal();
      fft_2d.cleanup();

      // If there was an error then throw an exception
      if (DeviceType == mt::e_device) {
        auto err = cudaGetLastError();
        if (err != cudaSuccess) {
          throw multem::Error(__FILE__, __LINE__, cudaGetErrorString(err));
        }
      }
    }

    /**
     * Run the multislice simulation with multi-region material GRF.
     */
    template <typename FloatType, mt::eDevice DeviceType>
    void run_multislice_internal(
      const mt::System_Configuration &system_conf,
      mt::Input_Multislice<FloatType> &input_multislice,
      mt::Output_Multislice<FloatType> &output_multislice,
      const MultiRegionMasker &masker) {

      // Set the system configuration
      input_multislice.system_conf = system_conf;

      // Open a stream
      mt::Stream<DeviceType> stream(system_conf.nstream);

      // Create the FFT object
      mt::FFT<FloatType, DeviceType> fft_2d;
      fft_2d.create_plan_2d(
        input_multislice.grid_2d.ny,
        input_multislice.grid_2d.nx,
        system_conf.nstream);

      // Setup the multislice simulation
      mt::Multislice<FloatType, DeviceType> multislice;
      multislice.set_input_data(&input_multislice, &stream, &fft_2d);

      // Compute the pixel sizes
      double x_pixel_size = input_multislice.atoms.l_x / input_multislice.grid_2d.nx;
      double y_pixel_size = input_multislice.atoms.l_y / input_multislice.grid_2d.ny;

      // Setup the multi-material potential approximation
      // NB: set_masker must be called before set_grid, because set_masker
      // recreates the region array and set_grid resizes the region buffers.
      MultiMaterialPotentialApproximation<FloatType, DeviceType> potential_function;
      potential_function.set_fft_2d(&fft_2d);
      potential_function.set_masker(masker);
      seed_from_masker(potential_function, masker);
      potential_function.set_grid(input_multislice.grid_2d);
      potential_function.set_pixel_size(x_pixel_size, y_pixel_size);
      multislice.set_potential_function(&potential_function);

      // Set the input data
      output_multislice.set_input_data(&input_multislice);

      // Perform the multislice simulation
      multislice(output_multislice);
      stream.synchronize();

      // Get the results
      output_multislice.gather();
      output_multislice.clean_temporal();
      fft_2d.cleanup();

      // If there was an error then throw an exception
      if (DeviceType == mt::e_device) {
        auto err = cudaGetLastError();
        if (err != cudaSuccess) {
          throw multem::Error(__FILE__, __LINE__, cudaGetErrorString(err));
        }
      }
    }

    /**
     * Run the multislice simulation
     * @param system_conf The system configuration
     * @param input_multislice The input object
     * @param output_multislice The output object
     */
    template <typename FloatType, mt::eDevice DeviceType>
    void run_projected_potential_internal(
      const mt::System_Configuration &system_conf,
      mt::Input_Multislice<FloatType> &input_multislice,
      mt::Output_Multislice<FloatType> &output_multislice,
      projected_potential_callback callback) {
  
      // Set the system configration    
      input_multislice.system_conf = system_conf;

      // Open a stream
      mt::Stream<DeviceType> stream(system_conf.nstream);

      // Setup the multislice simulation 
      mt::Projected_Potential<FloatType, DeviceType> projected_potential;
      projected_potential.set_input_data(&input_multislice, &stream);
      
      // Set the input data
      output_multislice.set_input_data(&input_multislice);

      // Perform the multislice simulation
      mt::Vector<FloatType, DeviceType> V(input_multislice.grid_2d.nxy());
      mt::Vector<FloatType, mt::e_host> V_host(input_multislice.grid_2d.nxy());
      for (auto islice = 0; islice < projected_potential.slicing.slice.size(); ++islice) {
        double z_0 = projected_potential.slicing.slice[islice].z_0;
        double z_e = projected_potential.slicing.slice[islice].z_e;
        projected_potential(islice, V);
        mt::fft2_shift(input_multislice.grid_2d, V);
        V_host.assign(V.begin(), V.end());
        MULTEM_ASSERT(input_multislice.grid_2d.nx >= 0);
        MULTEM_ASSERT(input_multislice.grid_2d.ny >= 0);
        callback(
            z_0, 
            z_e, 
            Image<double>(V_host.data(), 
              Image<double>::shape_type({
                (std::size_t)input_multislice.grid_2d.nx,
                (std::size_t)input_multislice.grid_2d.ny})));
      }

      // Syncronize stream
      stream.synchronize();
      
      // Get the results
      output_multislice.gather();
      output_multislice.clean_temporal();

      // If there was an error then throw an exception
      if (DeviceType == mt::e_device) {
        auto err = cudaGetLastError();
        if (err != cudaSuccess) {
          throw multem::Error(__FILE__, __LINE__, cudaGetErrorString(err));
        }
      }
    }
    
    /**
     * The random-field potential for a masker type: the ice approximation for a Masker, the
     * multi-material one for a MultiRegionMasker. Each is set up in the order its own simulate path
     * uses, so the projected potential is the one the exit wave sees.
     */
    template <typename MaskerType, typename FloatType, mt::eDevice DeviceType>
    struct potential_function_for;

    template <typename FloatType, mt::eDevice DeviceType>
    struct potential_function_for<Masker, FloatType, DeviceType> {
      using type = IcePotentialApproximation<FloatType, DeviceType>;
      static void setup(type &pf, const Masker &masker, mt::FFT<FloatType, DeviceType> *fft,
                        const mt::Grid_2d<FloatType> &grid, double px, double py) {
        pf.set_fft_2d(fft);
        pf.set_grid(grid);
        pf.set_masker(masker);
        seed_from_masker(pf, masker);
        pf.set_pixel_size(px, py);
      }
    };

    template <typename FloatType, mt::eDevice DeviceType>
    struct potential_function_for<MultiRegionMasker, FloatType, DeviceType> {
      using type = MultiMaterialPotentialApproximation<FloatType, DeviceType>;
      static void setup(type &pf, const MultiRegionMasker &masker, mt::FFT<FloatType, DeviceType> *fft,
                        const mt::Grid_2d<FloatType> &grid, double px, double py) {
        pf.set_fft_2d(fft);
        pf.set_masker(masker);
        seed_from_masker(pf, masker);
        pf.set_grid(grid);
        pf.set_pixel_size(px, py);
      }
    };

    /**
     * Run the multislice simulation
     * @param system_conf The system configuration
     * @param input_multislice The input object
     * @param output_multislice The output object
     */
    template <typename FloatType, mt::eDevice DeviceType, typename Masker>
    void run_projected_potential_internal(
      const mt::System_Configuration &system_conf,
      mt::Input_Multislice<FloatType> &input_multislice,
      mt::Output_Multislice<FloatType> &output_multislice,
      const Masker &masker,
      projected_potential_callback callback) {
  
      // Set the system configration    
      input_multislice.system_conf = system_conf;

      // Open a stream
      mt::Stream<DeviceType> stream(system_conf.nstream);

      // Create the FFT object
      mt::FFT<FloatType, DeviceType> fft_2d;
      fft_2d.create_plan_2d(
        input_multislice.grid_2d.ny, 
        input_multislice.grid_2d.nx, 
        system_conf.nstream);

      // Setup the multislice simulation 
      mt::Projected_Potential<FloatType, DeviceType> projected_potential;
      projected_potential.set_input_data(&input_multislice, &stream);
   
      // Compute the pixel sizes
      double x_pixel_size = input_multislice.atoms.l_x / input_multislice.grid_2d.nx;
      double y_pixel_size = input_multislice.atoms.l_y / input_multislice.grid_2d.ny;

      // Setup the random-field potential for this masker type
      using PF = potential_function_for<Masker, FloatType, DeviceType>;
      typename PF::type potential_function;
      PF::setup(potential_function, masker, &fft_2d, input_multislice.grid_2d, x_pixel_size, y_pixel_size);
      
      // Set the input data
      output_multislice.set_input_data(&input_multislice);

      // Perform the multislice simulation
      mt::Vector<FloatType, DeviceType> V(input_multislice.grid_2d.nxy());
      mt::Vector<FloatType, mt::e_host> V_host(input_multislice.grid_2d.nxy());
      for (auto islice = 0; islice < projected_potential.slicing.slice.size(); ++islice) {
        double z_0 = projected_potential.slicing.slice[islice].z_0;
        double z_e = projected_potential.slicing.slice[islice].z_e;
        projected_potential(islice, V);
        potential_function(z_0, z_e, V);  
        mt::fft2_shift(input_multislice.grid_2d, V);
        V_host.assign(V.begin(), V.end());
        MULTEM_ASSERT(input_multislice.grid_2d.nx >= 0);
        MULTEM_ASSERT(input_multislice.grid_2d.ny >= 0);
        callback(
            z_0, 
            z_e, 
            Image<double>(V_host.data(), 
              Image<double>::shape_type({
                (std::size_t)input_multislice.grid_2d.nx,
                (std::size_t)input_multislice.grid_2d.ny})));
      }

      // Syncronize stream
      stream.synchronize();
      
      // Get the results
      output_multislice.gather();
      output_multislice.clean_temporal();

      // If there was an error then throw an exception
      if (DeviceType == mt::e_device) {
        auto err = cudaGetLastError();
        if (err != cudaSuccess) {
          throw multem::Error(__FILE__, __LINE__, cudaGetErrorString(err));
        }
      }
    }
 
    /**
     * Convert the multem::SystemConfiguration object to a
     * mt::System_Configuration object
     * @param config The multem::SystemConfiguration object
     * @returns The mt::System_Configuration object
     */
    mt::System_Configuration read_system_configuration(const SystemConfiguration &config) {
      mt::System_Configuration system_conf;
      system_conf.device = detail::from_string<mt::eDevice>(config.device);
      system_conf.precision = detail::from_string<mt::ePrecision>(config.precision);
      system_conf.cpu_ncores = config.cpu_ncores;
      system_conf.cpu_nthread = config.cpu_nthread;
      system_conf.gpu_device = config.gpu_device;
      system_conf.gpu_nstream = config.gpu_nstream;
      system_conf.active = true;
      system_conf.validate_parameters();
      system_conf.set_device();
      return system_conf;
    } 
    
    /**
     * Convert the multem::Input object to a
     * mt::Input_Multislice object
     * @param config The multem::Input object
     * @returns The mt::Input_Multislice object
     */
    template <typename FloatType>
    mt::Input_Multislice<FloatType> read_input_multislice(
        const Input &input,
        bool full = true) {
      mt::Input_Multislice<FloatType> input_multislice;

      // Simulation type
      input_multislice.simulation_type = 
        detail::from_string<mt::eTEM_Sim_Type>(input.simulation_type);
      input_multislice.interaction_model = 
        detail::from_string<mt::eElec_Spec_Int_Model>(input.interaction_model);
      input_multislice.potential_type = 
        detail::from_string<mt::ePotential_Type>(input.potential_type);
      input_multislice.operation_mode = 
        detail::from_string<mt::eOperation_Mode>(input.operation_mode);
      input_multislice.reverse_multislice = input.reverse_multislice;

      // Electron-Phonon interaction model
      input_multislice.pn_model = detail::from_string<mt::ePhonon_Model>(input.pn_model);
      input_multislice.pn_coh_contrib = input.pn_coh_contrib;
      input_multislice.pn_single_conf = input.pn_single_conf;
      input_multislice.pn_nconf = input.pn_nconf;
      input_multislice.pn_dim.set(input.pn_dim);
      input_multislice.pn_seed = input.pn_seed;
      input_multislice.static_B_factor = input.static_B_factor;

      // Specimen
      bool pbc_xy = true;

      // Set the specimen
      if (input_multislice.is_specimen_required())
      {
        // Set the amorphous layer information
        mt::Vector<mt::Amorp_Lay_Info<FloatType>, mt::e_host> amorp_lay_info;
        for (auto item : input.spec_amorp) {
          mt::Amorp_Lay_Info<FloatType> value;
          value.z_0 = item.z_0;
          value.z_e = item.z_e;
          value.dz = item.dz;
          amorp_lay_info.push_back(value);
        }

        if (full) {
          input_multislice.atoms.set_crystal_parameters(
              input.spec_cryst_na, 
              input.spec_cryst_nb, 
              input.spec_cryst_nc, 
              input.spec_cryst_a, 
              input.spec_cryst_b, 
              input.spec_cryst_c, 
              input.spec_cryst_x0, 
              input.spec_cryst_y0);
          input_multislice.atoms.set_amorphous_parameters(amorp_lay_info);
          input_multislice.atoms.l_x = input.spec_lx;
          input_multislice.atoms.l_y = input.spec_ly;
          input_multislice.atoms.l_z = input.spec_lz;
          input_multislice.atoms.dz = input.spec_dz;
          input_multislice.atoms.resize(input.spec_atoms.size());
          for(auto i = 0; i < input.spec_atoms.size(); ++i) {
            input_multislice.atoms.Z[i] = input.spec_atoms[i].element;
            input_multislice.atoms.x[i] = input.spec_atoms[i].x;
            input_multislice.atoms.y[i] = input.spec_atoms[i].y;
            input_multislice.atoms.z[i] = input.spec_atoms[i].z;
            input_multislice.atoms.sigma[i] = input.spec_atoms[i].sigma;
            input_multislice.atoms.occ[i] = input.spec_atoms[i].occupancy;
            input_multislice.atoms.region[i] = abs(input.spec_atoms[i].region);
            input_multislice.atoms.charge[i] = input.spec_atoms[i].charge;
          }
          input_multislice.atoms.get_statistic();
        }

        // Specimen rotation
        input_multislice.spec_rot_theta = input.spec_rot_theta*mt::c_deg_2_rad;
        input_multislice.spec_rot_u0 = mt::r3d<FloatType>(
            input.spec_rot_u0[0],
            input.spec_rot_u0[1],
            input.spec_rot_u0[2]);
        input_multislice.spec_rot_u0.normalized();
        input_multislice.spec_rot_center_type =
          detail::from_string<mt::eRot_Point_Type>(input.spec_rot_center_type);
        input_multislice.spec_rot_center_p = mt::r3d<FloatType>(
            input.spec_rot_center_p[0],
            input.spec_rot_center_p[1],
            input.spec_rot_center_p[2]);

        // Specimen thickness
        input_multislice.thick_type = detail::from_string<mt::eThick_Type>(input.thick_type);
        if (!input_multislice.is_whole_spec() && full) {
          input_multislice.thick.assign(input.thick.begin(), input.thick.end());
          MULTEM_ASSERT(input_multislice.thick.size() > 0);
        }

        // Potential slicing
        input_multislice.potential_slicing =
          detail::from_string<mt::ePotential_Slicing>(input.potential_slicing);
      }

      // XY sampling
      input_multislice.grid_2d.set_input_data(
          input.nx, 
          input.ny, 
          input.spec_lx, 
          input.spec_ly, 
          input.spec_dz, 
          input.bwl, 
          pbc_xy);

      // Incident wave
      input_multislice.set_incident_wave_type(detail::from_string<mt::eIncident_Wave_Type>(input.iw_type));

      if (input_multislice.is_user_define_wave() && full) {
        input_multislice.iw_psi.assign(
            input.iw_psi.begin(),
            input.iw_psi.end());
      }

      // read iw_x and iw_y
      int n_iw_xy = std::min(input.iw_x.size(), input.iw_y.size());
      input_multislice.iw_x.assign(input.iw_x.begin(), input.iw_x.begin() + n_iw_xy);
      input_multislice.iw_y.assign(input.iw_y.begin(), input.iw_y.begin() + n_iw_xy);

      // Microscope parameter
      input_multislice.E_0 = input.E_0;
      input_multislice.theta = input.theta*mt::c_deg_2_rad;
      input_multislice.phi = input.phi*mt::c_deg_2_rad;

      // Illumination model
      input_multislice.illumination_model =
        detail::from_string<mt::eIllumination_Model>(input.illumination_model);
      input_multislice.temporal_spatial_incoh =
        detail::from_string<mt::eTemporal_Spatial_Incoh>(input.temporal_spatial_incoh);

      // Condenser lens
      input_multislice.cond_lens.m = input.cond_lens_m;
      input_multislice.cond_lens.c_10 = input.cond_lens_c_10;
      input_multislice.cond_lens.c_12 = input.cond_lens_c_12;
      input_multislice.cond_lens.phi_12 = input.cond_lens_phi_12*mt::c_deg_2_rad;
      input_multislice.cond_lens.c_21 = input.cond_lens_c_21;
      input_multislice.cond_lens.phi_21 = input.cond_lens_phi_21*mt::c_deg_2_rad;
      input_multislice.cond_lens.c_23 = input.cond_lens_c_23;
      input_multislice.cond_lens.phi_23 = input.cond_lens_phi_23*mt::c_deg_2_rad;
      input_multislice.cond_lens.c_30 = input.cond_lens_c_30*mt::c_mm_2_Angs; 
      input_multislice.cond_lens.c_32 = input.cond_lens_c_32; 
      input_multislice.cond_lens.phi_32 = input.cond_lens_phi_32*mt::c_deg_2_rad;
      input_multislice.cond_lens.c_34 = input.cond_lens_c_34;
      input_multislice.cond_lens.phi_34 = input.cond_lens_phi_34*mt::c_deg_2_rad;
      input_multislice.cond_lens.c_41 = input.cond_lens_c_41;
      input_multislice.cond_lens.phi_41 = input.cond_lens_phi_41*mt::c_deg_2_rad;
      input_multislice.cond_lens.c_43 = input.cond_lens_c_43;
      input_multislice.cond_lens.phi_43 = input.cond_lens_phi_43*mt::c_deg_2_rad;
      input_multislice.cond_lens.c_45 = input.cond_lens_c_45;
      input_multislice.cond_lens.phi_45 = input.cond_lens_phi_45*mt::c_deg_2_rad;
      input_multislice.cond_lens.c_50 = input.cond_lens_c_50*mt::c_mm_2_Angs;
      input_multislice.cond_lens.c_52 = input.cond_lens_c_52;
      input_multislice.cond_lens.phi_52 = input.cond_lens_phi_52*mt::c_deg_2_rad;
      input_multislice.cond_lens.c_54 = input.cond_lens_c_54;
      input_multislice.cond_lens.phi_54 = input.cond_lens_phi_54*mt::c_deg_2_rad;
      input_multislice.cond_lens.c_56 = input.cond_lens_c_56;
      input_multislice.cond_lens.phi_56 = input.cond_lens_phi_56*mt::c_deg_2_rad;
      input_multislice.cond_lens.inner_aper_ang = input.cond_lens_inner_aper_ang*mt::c_mrad_2_rad;
      input_multislice.cond_lens.outer_aper_ang = input.cond_lens_outer_aper_ang*mt::c_mrad_2_rad;

      // defocus spread function
      /* input_multislice.cond_lens.set_ti_sigma(input.cond_lens_ti_sigma); */
      /* input_multislice.cond_lens.dsf_npoints = input.cond_lens_dsf_npoints; */

	    input_multislice.cond_lens.ti_a = input.cond_lens_ti_a;
	    input_multislice.cond_lens.ti_sigma = input.cond_lens_ti_sigma;
	    input_multislice.cond_lens.ti_beta = input.cond_lens_ti_beta;
	    input_multislice.cond_lens.ti_npts = input.cond_lens_ti_npts;

      // source spread function
      /* input_multislice.cond_lens.set_si_sigma(input.cond_lens_si_sigma); */
      /* input_multislice.cond_lens.ssf_npoints = input.cond_lens_ssf_npoints; */

      input_multislice.cond_lens.si_a = input.cond_lens_si_a;
      input_multislice.cond_lens.si_sigma = input.cond_lens_si_sigma;
      input_multislice.cond_lens.si_beta = input.cond_lens_si_beta;
      input_multislice.cond_lens.si_rad_npts = input.cond_lens_si_rad_npts;
      input_multislice.cond_lens.si_azm_npts = input.cond_lens_si_azm_npts; 

      // zero defocus reference
      input_multislice.cond_lens.zero_defocus_type = 
        detail::from_string<mt::eZero_Defocus_Type>(input.cond_lens_zero_defocus_type);
      input_multislice.cond_lens.zero_defocus_plane = input.cond_lens_zero_defocus_plane;
      input_multislice.cond_lens.set_input_data(input_multislice.E_0, input_multislice.grid_2d);

      // Objective lens
      input_multislice.obj_lens.m = input.obj_lens_m;
      input_multislice.obj_lens.c_10 = input.obj_lens_c_10;
      input_multislice.obj_lens.c_12 = input.obj_lens_c_12;
      input_multislice.obj_lens.phi_12 = input.obj_lens_phi_12*mt::c_deg_2_rad;
      input_multislice.obj_lens.c_21 = input.obj_lens_c_21;
      input_multislice.obj_lens.phi_21 = input.obj_lens_phi_21*mt::c_deg_2_rad;
      input_multislice.obj_lens.c_23 = input.obj_lens_c_23;
      input_multislice.obj_lens.phi_23 = input.obj_lens_phi_23*mt::c_deg_2_rad;
      input_multislice.obj_lens.c_30 = input.obj_lens_c_30*mt::c_mm_2_Angs;
      input_multislice.obj_lens.c_32 = input.obj_lens_c_32;
      input_multislice.obj_lens.phi_32 = input.obj_lens_phi_32*mt::c_deg_2_rad;
      input_multislice.obj_lens.c_34 = input.obj_lens_c_34;
      input_multislice.obj_lens.phi_34 = input.obj_lens_phi_34*mt::c_deg_2_rad;
      input_multislice.obj_lens.c_41 = input.obj_lens_c_41;
      input_multislice.obj_lens.phi_41 = input.obj_lens_phi_41*mt::c_deg_2_rad;
      input_multislice.obj_lens.c_43 = input.obj_lens_c_43;
      input_multislice.obj_lens.phi_43 = input.obj_lens_phi_43*mt::c_deg_2_rad;
      input_multislice.obj_lens.c_45 = input.obj_lens_c_45;
      input_multislice.obj_lens.phi_45 = input.obj_lens_phi_45*mt::c_deg_2_rad;
      input_multislice.obj_lens.c_50 = input.obj_lens_c_50*mt::c_mm_2_Angs;
      input_multislice.obj_lens.c_52 = input.obj_lens_c_52;
      input_multislice.obj_lens.phi_52 = input.obj_lens_phi_52*mt::c_deg_2_rad;
      input_multislice.obj_lens.c_54 = input.obj_lens_c_54;
      input_multislice.obj_lens.phi_54 = input.obj_lens_phi_54*mt::c_deg_2_rad;
      input_multislice.obj_lens.c_56 = input.obj_lens_c_56;
      input_multislice.obj_lens.phi_56 = input.obj_lens_phi_56*mt::c_deg_2_rad;
      input_multislice.obj_lens.inner_aper_ang = input.obj_lens_inner_aper_ang*mt::c_mrad_2_rad;
      input_multislice.obj_lens.outer_aper_ang = input.obj_lens_outer_aper_ang*mt::c_mrad_2_rad;

      // defocus spread function
      /* input_multislice.obj_lens.set_ti_sigma(input.obj_lens_ti_sigma); */
      /* input_multislice.obj_lens.dsf_npoints = input.obj_lens_dsf_npoints; */
      
      input_multislice.obj_lens.si_a = input.cond_lens_si_a;
      input_multislice.obj_lens.si_sigma = input.cond_lens_si_sigma;
      input_multislice.obj_lens.si_beta = input.cond_lens_si_beta;
      input_multislice.obj_lens.si_rad_npts = input.cond_lens_si_rad_npts;
      input_multislice.obj_lens.si_azm_npts = input.cond_lens_si_azm_npts; 
      
      input_multislice.obj_lens.ti_a = input.obj_lens_ti_a;
	    input_multislice.obj_lens.ti_sigma = input.obj_lens_ti_sigma;
	    input_multislice.obj_lens.ti_beta = input.obj_lens_ti_beta;
	    input_multislice.obj_lens.ti_npts = input.obj_lens_ti_npts;

      // source spread function
      /* input_multislice.obj_lens.set_si_sigma(input_multislice.cond_lens.si_sigma); */
      /* input_multislice.obj_lens.ssf_npoints = input_multislice.cond_lens.ssf_npoints; */
	    
      // zero defocus reference
      input_multislice.obj_lens.zero_defocus_type = 
        detail::from_string<mt::eZero_Defocus_Type>(input.obj_lens_zero_defocus_type);
      input_multislice.obj_lens.zero_defocus_plane = input.obj_lens_zero_defocus_plane;
      input_multislice.obj_lens.set_input_data(input_multislice.E_0, input_multislice.grid_2d);

      // Set the phase shift
      input_multislice.phase_shift = input.phase_shift;

      // ISTEM/STEM 
      if (input_multislice.is_scanning()) {
        input_multislice.scanning.type = detail::from_string<mt::eScanning_Type>(input.scanning_type);
        input_multislice.scanning.pbc = input.scanning_periodic;
        input_multislice.scanning.spxs = input.scanning_square_pxs;
        input_multislice.scanning.ns = input.scanning_ns;
        input_multislice.scanning.x0 = input.scanning_x0;
        input_multislice.scanning.y0 = input.scanning_y0;
        input_multislice.scanning.xe = input.scanning_xe;
        input_multislice.scanning.ye = input.scanning_ye;
        input_multislice.scanning.set_grid();
      }

      if (input_multislice.is_STEM()) {
        FloatType lambda = mt::get_lambda(input_multislice.E_0);
        input_multislice.detector.type = detail::from_string<mt::eDetector_Type>(input.detector.type);

        switch (input_multislice.detector.type) {
        case mt::eDT_Circular: 
        {
          int ndetector = input.detector.cir.size();
          if (ndetector > 0) {
            input_multislice.detector.resize(ndetector);
            for (auto i = 0; i < input_multislice.detector.size(); i++) {
              auto inner_ang = input.detector.cir[i].inner_ang*mt::c_mrad_2_rad;
              auto outer_ang = input.detector.cir[i].outer_ang*mt::c_mrad_2_rad;
              input_multislice.detector.g_inner[i] = std::sin(inner_ang) / lambda;
              input_multislice.detector.g_outer[i] = std::sin(outer_ang) / lambda;
            }
          }
        }
        break;
        case mt::eDT_Radial:
        {
          int ndetector = input.detector.radial.size();
          if (ndetector > 0) {
            input_multislice.detector.resize(ndetector);
            for (auto i = 0; i < input_multislice.detector.size(); i++) {
              input_multislice.detector.fx[i].assign(
                  input.detector.radial[i].fx.begin(),
                  input.detector.radial[i].fx.end());
            }
          }
        }
        break;
        case mt::eDT_Matrix:
        {
          int ndetector = input.detector.matrix.size();
          if (ndetector > 0) {
            input_multislice.detector.resize(ndetector);
            for (auto i = 0; i < input_multislice.detector.size(); i++) {
              input_multislice.detector.fR[i].assign(
                  input.detector.matrix[i].fR.begin(),
                  input.detector.matrix[i].fR.end());
            }
          }
        }
        break;
        };
      } else if (input_multislice.is_PED()) {
        input_multislice.theta = input.ped_theta*mt::c_deg_2_rad;
        input_multislice.nrot = input.ped_nrot;
      } else if (input_multislice.is_HCTEM()) {
        input_multislice.theta = input.hci_theta*mt::c_deg_2_rad;
        input_multislice.nrot = input.hci_nrot;
      } else if (input_multislice.is_EELS()) {
        input_multislice.eels_fr.set_input_data(
          mt::eS_Reciprocal, 
          input_multislice.E_0, 
          input.eels_E_loss * mt::c_eV_2_keV, 
          input.eels_m_selection, 
          input.eels_collection_angle * mt::c_mrad_2_rad, 
          detail::from_string<mt::eChannelling_Type>(input.eels_channelling_type),
          input.eels_Z);
      } else if (input_multislice.is_EFTEM()) {
        input_multislice.eels_fr.set_input_data(
            mt::eS_Real, 
            input_multislice.E_0, 
            input.eftem_E_loss * mt::c_eV_2_keV, 
            input.eftem_m_selection, 
            input.eftem_collection_angle * mt::c_mrad_2_rad, 
            detail::from_string<mt::eChannelling_Type>(input.eftem_channelling_type),
            input.eftem_Z);
      }

      // Select the output region
      input_multislice.output_area.ix_0 = input.output_area_ix_0-1;
      input_multislice.output_area.iy_0 = input.output_area_iy_0-1;
      input_multislice.output_area.ix_e = input.output_area_ix_e-1;
      input_multislice.output_area.iy_e = input.output_area_iy_e-1;

      // Validate the input parameters
      input_multislice.validate_parameters();
      /* mt::print(input_multislice); */
      return input_multislice;
    }

    /**
     * Convert the mt::Output_Multislice object to a multem::Output object
     * @param output_multislice The mt::Output_Multislice object
     * @returns Output The multem::Output object
     */
    template <typename FloatType>
    Output write_output_multislice(const mt::Output_Multislice<FloatType> &output_multislice) {
      
      // Set some general properties
      Output result;
      result.dx = output_multislice.dx;
      result.dy = output_multislice.dy;
      result.x.assign(output_multislice.x.begin(), output_multislice.x.end());
      result.y.assign(output_multislice.y.begin(), output_multislice.y.end());
      result.thick.assign(output_multislice.thick.begin(), output_multislice.thick.end());
      result.data.resize(output_multislice.thick.size());

      // Write the output data
      if (output_multislice.is_STEM() || output_multislice.is_EELS()) {
        std::size_t nx = (output_multislice.scanning.is_line()) ? 1 : output_multislice.nx;
        std::size_t ny = output_multislice.ny;
        for (auto i = 0; i < output_multislice.thick.size(); ++i) {
          result.data[i].image_tot.resize(output_multislice.ndetector);
          if (output_multislice.pn_coh_contrib) {
            result.data[i].image_coh.resize(output_multislice.ndetector);
          }
          for (auto j = 0; j < output_multislice.ndetector; ++j) {
            result.data[i].image_tot[j] = Image<double>(
                output_multislice.image_tot[i].image[j].data(), 
                  Image<double>::shape_type({ nx, ny }));
            if (output_multislice.pn_coh_contrib) {
              result.data[i].image_coh[j] = Image<double>(
                  output_multislice.image_coh[i].image[j].data(), 
                    Image<double>::shape_type({ nx, ny }));
            }
          }
        }
      } else if (output_multislice.is_EWFS_EWRS()) {
        for (auto i = 0; i < output_multislice.thick.size(); ++i) {
          if (!output_multislice.is_EWFS_EWRS_SC()) {
            if (result.data.size() == output_multislice.m2psi_tot.size()) {
              result.data[i].m2psi_tot = Image<double>(
                  output_multislice.m2psi_tot[i].data(), 
                    Image<double>::shape_type({
                      (std::size_t) output_multislice.nx,
                      (std::size_t) output_multislice.ny}));
            }
          }
          if (result.data.size() == output_multislice.psi_coh.size()) {
            result.data[i].psi_coh = Image< std::complex<double> >(
                output_multislice.psi_coh[i].data(), 
                Image< std::complex<double> >::shape_type({
                  (std::size_t) output_multislice.nx,
                  (std::size_t) output_multislice.ny}));
          }
        }
      } else {
        for (auto i = 0; i < output_multislice.thick.size(); ++i) {
          if (result.data.size() == output_multislice.m2psi_tot.size()) {
            result.data[i].m2psi_tot = Image<double>(
                output_multislice.m2psi_tot[i].data(), 
                  Image<double>::shape_type({
                    (std::size_t) output_multislice.nx,
                    (std::size_t) output_multislice.ny}));
          }
          if (output_multislice.pn_coh_contrib) {
            if (result.data.size() == output_multislice.m2psi_coh.size()) {
              result.data[i].m2psi_coh = Image<double>(
                  output_multislice.m2psi_coh[i].data(), 
                  Image<double>::shape_type({
                    (std::size_t) output_multislice.nx,
                    (std::size_t) output_multislice.ny}));
            }
          }
        }
      }

      if (result.data.size() == output_multislice.V.size()) {
        for (auto i = 0; i < output_multislice.V.size(); ++i) {
          MULTEM_ASSERT(output_multislice.V[i].size() == output_multislice.nx*output_multislice.ny);
          result.data[i].V = Image<double>(
                output_multislice.V[i].data(), 
                Image<double>::shape_type({
                  (std::size_t) output_multislice.nx,
                  (std::size_t) output_multislice.ny}));
        }
      }


      // Return the result
      return result;
    }
  }

  /**
   * Run the multislice simulation. 
   * @param config The system configuration
   * @param input The input object
   * @returns The output results
   */
  template <typename FloatType, mt::eDevice DeviceType>
  Output run_multislice(SystemConfiguration config, Input input) {

    // Initialise the system configuration and input structures 
    auto system_conf = detail::read_system_configuration(config);
    auto input_multislice = detail::read_input_multislice<FloatType>(input);
    input_multislice.system_conf = system_conf;

    // Create the output structure
    mt::Output_Multislice<FloatType> output_multislice;
   
    // Run the simulation 
    detail::run_multislice_internal<FloatType, DeviceType>(
      system_conf, input_multislice, output_multislice);

    // Return the output struct
    return detail::write_output_multislice(output_multislice);
  }
  
  /**
   * Run the multislice simulation. 
   * @param config The system configuration
   * @param input The input object
   * @returns The output results
   */
  template <typename FloatType, mt::eDevice DeviceType, typename Masker>
  Output run_multislice(SystemConfiguration config, Input input, const Masker &masker) {

    // Add a couple of hydrogen atoms at extreme z values. This is needed
    // because if multem has no atoms it gets confused. By adding an atom at
    // the minimum and maximum Z locations we make multem create slices between
    // those two atoms.
    input.spec_atoms.push_back(Atom(1, 0, 0, masker.zmin(), 0, 1, 0, 0));
    input.spec_atoms.push_back(Atom(1, 0, 0, masker.zmax(), 0, 1, 0, 0));

    // Initialise the system configuration and input structures 
    auto system_conf = detail::read_system_configuration(config);
    auto input_multislice = detail::read_input_multislice<FloatType>(input);
    input_multislice.system_conf = system_conf;

    // Create the output structure
    mt::Output_Multislice<FloatType> output_multislice;
   
    // Run the simulation 
    detail::run_multislice_internal<FloatType, DeviceType, Masker>(
      system_conf, input_multislice, output_multislice, masker);

    // Return the output struct
    return detail::write_output_multislice(output_multislice);
  }

  Output simulate(SystemConfiguration config, Input input) {
    Output result;
    if (config.device == "host" && config.precision == "float") {
      result = run_multislice<float, mt::e_host>(config, input);
    } else if (config.device == "host" && config.precision == "double") {
      result = run_multislice<double, mt::e_host>(config, input);
#ifndef MULTEM_NO_CUDA
    } else if (config.device == "device" && config.precision == "float") {
      result = run_multislice<float, mt::e_device>(config, input);
    } else if (config.device == "device" && config.precision == "double") {
      result = run_multislice<double, mt::e_device>(config, input);
#endif
    } else {
      MULTEM_ASSERT(config.device == "host" || config.device == "device");
      MULTEM_ASSERT(config.precision == "float" || config.precision == "double");
    }
    return result;
  }

  Output simulate_with_ice_approximation(
      SystemConfiguration config, 
      Input input, 
      const Masker &masker) {
    Output result;
    if (config.device == "host" && config.precision == "float") {
      result = run_multislice<float, mt::e_host, Masker>(config, input, masker);
    } else if (config.device == "host" && config.precision == "double") {
      result = run_multislice<double, mt::e_host, Masker>(config, input, masker);
#ifndef MULTEM_NO_CUDA
    } else if (config.device == "device" && config.precision == "float") {
      result = run_multislice<float, mt::e_device, Masker>(config, input, masker);
    } else if (config.device == "device" && config.precision == "double") {
      result = run_multislice<double, mt::e_device, Masker>(config, input, masker);
#endif
    } else {
      MULTEM_ASSERT(config.device == "host" || config.device == "device");
      MULTEM_ASSERT(config.precision == "float" || config.precision == "double");
    }
    return result;
  }

  /**
   * Run the multislice simulation with multi-region material GRF.
   */
  template <typename FloatType, mt::eDevice DeviceType>
  Output run_multislice(
      SystemConfiguration config,
      Input input,
      const MultiRegionMasker &masker) {

    // Add sentinel atoms at z extremes (same pattern as single-masker variant)
    input.spec_atoms.push_back(Atom(1, 0, 0, masker.zmin(), 0, 1, 0, 0));
    input.spec_atoms.push_back(Atom(1, 0, 0, masker.zmax(), 0, 1, 0, 0));

    // Initialise the system configuration and input structures
    auto system_conf = detail::read_system_configuration(config);
    auto input_multislice = detail::read_input_multislice<FloatType>(input);
    input_multislice.system_conf = system_conf;

    // Create the output structure
    mt::Output_Multislice<FloatType> output_multislice;

    // Run the simulation
    detail::run_multislice_internal<FloatType, DeviceType>(
      system_conf, input_multislice, output_multislice, masker);

    // Return the output struct
    return detail::write_output_multislice(output_multislice);
  }

  /**
   * Run the multislice simulation with MultiRegionMasker.
   *
   * This is a separate overload (not using the generic run_multislice template)
   * to ensure the MultiRegionMasker-specific run_multislice_internal overload
   * (which uses MultiMaterialPotentialApproximation) is called, rather than the
   * generic template version (which uses IcePotentialApproximation).
   */
  template <typename FloatType, mt::eDevice DeviceType>
  Output run_multislice_multi_material(
      SystemConfiguration config, Input input, const MultiRegionMasker &masker) {

    // Add hydrogen atoms at z extremes so MULTEM creates slices
    input.spec_atoms.push_back(Atom(1, 0, 0, masker.zmin(), 0, 1, 0, 0));
    input.spec_atoms.push_back(Atom(1, 0, 0, masker.zmax(), 0, 1, 0, 0));

    auto system_conf = detail::read_system_configuration(config);
    auto input_multislice = detail::read_input_multislice<FloatType>(input);
    input_multislice.system_conf = system_conf;

    mt::Output_Multislice<FloatType> output_multislice;

    // Calls the MultiRegionMasker-specific overload (2 template params)
    detail::run_multislice_internal<FloatType, DeviceType>(
      system_conf, input_multislice, output_multislice, masker);

    return detail::write_output_multislice(output_multislice);
  }

  Output simulate_with_multi_material(
      SystemConfiguration config,
      Input input,
      const MultiRegionMasker &masker) {
    Output result;
    if (config.device == "host" && config.precision == "float") {
      result = run_multislice_multi_material<float, mt::e_host>(config, input, masker);
    } else if (config.device == "host" && config.precision == "double") {
      result = run_multislice_multi_material<double, mt::e_host>(config, input, masker);
#ifndef MULTEM_NO_CUDA
    } else if (config.device == "device" && config.precision == "float") {
      result = run_multislice_multi_material<float, mt::e_device>(config, input, masker);
    } else if (config.device == "device" && config.precision == "double") {
      result = run_multislice_multi_material<double, mt::e_device>(config, input, masker);
#endif
    } else {
      MULTEM_ASSERT(config.device == "host" || config.device == "device");
      MULTEM_ASSERT(config.precision == "float" || config.precision == "double");
    }
    return result;
  }

  /**
   * Run the multislice simulation.
   * @param config The system configuration
   * @param input The input object
   * @returns The output results
   */
  template <typename FloatType, mt::eDevice DeviceType>
  Output run_projected_potential(
      SystemConfiguration config,
      Input input,
      projected_potential_callback callback) {

    // Initialise the system configuration and input structures 
    auto system_conf = detail::read_system_configuration(config);
    auto input_multislice = detail::read_input_multislice<FloatType>(input);
    input_multislice.system_conf = system_conf;

    // Create the output structure
    mt::Output_Multislice<FloatType> output_multislice;
   
    // Run the simulation 
    detail::run_projected_potential_internal<FloatType, DeviceType>(
      system_conf, input_multislice, output_multislice, callback);

    // Return the output struct
    return detail::write_output_multislice(output_multislice);
  }

  /**
   * Run the multislice simulation. 
   * @param config The system configuration
   * @param input The input object
   * @returns The output results
   */
  template <typename FloatType, mt::eDevice DeviceType, typename Masker>
  Output run_projected_potential(
      SystemConfiguration config, 
      Input input,
      const Masker &masker,
      projected_potential_callback callback) {
    
    // Add a couple of hydrogen atoms at extreme z values. This is needed
    // because if multem has no atoms it gets confused. By adding an atom at
    // the minimum and maximum Z locations we make multem create slices between
    // those two atoms.
    input.spec_atoms.push_back(Atom(1, 0, 0, masker.zmin(), 0, 1, 0, 0));
    input.spec_atoms.push_back(Atom(1, 0, 0, masker.zmax(), 0, 1, 0, 0));

    // Initialise the system configuration and input structures 
    auto system_conf = detail::read_system_configuration(config);
    auto input_multislice = detail::read_input_multislice<FloatType>(input);
    input_multislice.system_conf = system_conf;

    // Create the output structure
    mt::Output_Multislice<FloatType> output_multislice;
   
    // Run the simulation 
    detail::run_projected_potential_internal<FloatType, DeviceType, Masker>(
      system_conf, input_multislice, output_multislice, masker, callback);

    // Return the output struct
    return detail::write_output_multislice(output_multislice);
  }

  Output compute_projected_potential(
      SystemConfiguration config, 
      Input input,
      projected_potential_callback callback) {
    Output result;
    if (config.device == "host" && config.precision == "float") {
      result = run_projected_potential<float, mt::e_host>(config, input, callback);
    } else if (config.device == "host" && config.precision == "double") {
      result = run_projected_potential<double, mt::e_host>(config, input, callback);
#ifndef MULTEM_NO_CUDA
    } else if (config.device == "device" && config.precision == "float") {
      result = run_projected_potential<float, mt::e_device>(config, input, callback);
    } else if (config.device == "device" && config.precision == "double") {
      result = run_projected_potential<double, mt::e_device>(config, input, callback);
#endif
    } else {
      MULTEM_ASSERT(config.device == "host" || config.device == "device");
      MULTEM_ASSERT(config.precision == "float" || config.precision == "double");
    }
    return result;
  }
  
  Output compute_projected_potential_with_ice_approximation(
      SystemConfiguration config, 
      Input input,
      const Masker &masker,
      projected_potential_callback callback) {
    Output result;
    if (config.device == "host" && config.precision == "float") {
      result = run_projected_potential<float, mt::e_host, Masker>(config, input, masker, callback);
    } else if (config.device == "host" && config.precision == "double") {
      result = run_projected_potential<double, mt::e_host, Masker>(config, input, masker, callback);
#ifndef MULTEM_NO_CUDA
    } else if (config.device == "device" && config.precision == "float") {
      result = run_projected_potential<float, mt::e_device, Masker>(config, input, masker, callback);
    } else if (config.device == "device" && config.precision == "double") {
      result = run_projected_potential<double, mt::e_device, Masker>(config, input, masker, callback);
#endif
    } else {
      MULTEM_ASSERT(config.device == "host" || config.device == "device");
      MULTEM_ASSERT(config.precision == "float" || config.precision == "double");
    }
    return result;
  }

  Output compute_projected_potential_with_multi_material(
      SystemConfiguration config,
      Input input,
      const MultiRegionMasker &masker,
      projected_potential_callback callback) {
    Output result;
    if (config.device == "host" && config.precision == "float") {
      result = run_projected_potential<float, mt::e_host, MultiRegionMasker>(config, input, masker, callback);
    } else if (config.device == "host" && config.precision == "double") {
      result = run_projected_potential<double, mt::e_host, MultiRegionMasker>(config, input, masker, callback);
#ifndef MULTEM_NO_CUDA
    } else if (config.device == "device" && config.precision == "float") {
      result = run_projected_potential<float, mt::e_device, MultiRegionMasker>(config, input, masker, callback);
    } else if (config.device == "device" && config.precision == "double") {
      result = run_projected_potential<double, mt::e_device, MultiRegionMasker>(config, input, masker, callback);
#endif
    } else {
      MULTEM_ASSERT(config.device == "host" || config.device == "device");
      MULTEM_ASSERT(config.precision == "float" || config.precision == "double");
    }
    return result;
  }

  std::vector<double> compute_fraction_engine(
      const Masker &masker, double zs, double ze, int samples, bool device) {
    std::vector<double> out(masker.image_size());
    if (device) {
#ifndef MULTEM_NO_CUDA
      mt::Vector<double, mt::e_device> f(masker.image_size());
      mt::Vector<bool, mt::e_device> b(masker.image_size());
      detail::masker_compute_fraction(masker, zs, ze, samples, f, b);
      mt::Vector<double, mt::e_host> h(f.size());
      thrust::copy(f.begin(), f.end(), h.begin());
      std::copy(h.begin(), h.end(), out.begin());
#else
      MULTEM_ASSERT(!device);
#endif
    } else {
      mt::Vector<double, mt::e_host> f(masker.image_size());
      mt::Vector<bool, mt::e_host> b(masker.image_size());
      detail::masker_compute_fraction(masker, zs, ze, samples, f, b);
      std::copy(f.begin(), f.end(), out.begin());
    }
    return out;
  }

  std::vector<int> compute_mask_engine(const Masker &masker, double zs, double ze, bool device) {
    // the mask the simulation uses for one slice, on the host or in device memory (for tests)
    std::vector<int> out(masker.image_size());
    if (device) {
#ifndef MULTEM_NO_CUDA
      mt::Vector<bool, mt::e_device> m(masker.image_size());
      detail::masker_compute_mask(masker, zs, ze, m.begin());
      mt::Vector<bool, mt::e_host> h(m.size());
      thrust::copy(m.begin(), m.end(), h.begin());
      std::copy(h.begin(), h.end(), out.begin());
#else
      MULTEM_ASSERT(!device);
#endif
    } else {
      mt::Vector<bool, mt::e_host> h(masker.image_size());
      detail::masker_compute_mask(masker, zs, ze, h.begin());
      std::copy(h.begin(), h.end(), out.begin());
    }
    return out;
  }

  std::tuple<double, double, bool> ice_slice_statistics(
      const IceParameters &ice, double x_pixel_size, double y_pixel_size, double thickness) {
    // the mean and sigma the engine gives one slice of this material, from the same members the
    // simulation calls (band-limited switch, spectrum table, m0 and v0 included)
    MULTEM_ASSERT(thickness > 0 && x_pixel_size > 0 && y_pixel_size > 0);
    Masker masker(1, 1, x_pixel_size);
    masker.set_ice_parameters(ice);
    detail::IcePotentialApproximation<double, mt::e_host> pf;
    pf.set_masker(masker);
    pf.set_pixel_size(x_pixel_size, y_pixel_size);
    double n = thickness * pf.compute_number_density_of_water(ice.density);
    return std::make_tuple(pf.compute_mean(n), pf.compute_sigma(n), detail::band_limited_potential());
  }

  Image< std::complex<double> > compute_ctf(
      SystemConfiguration config, 
      Input input) {

    // Initialise the system configuration and input structures 
    config.device = "host";
    config.precision = "double";
    auto system_conf = detail::read_system_configuration(config);
    auto input_multislice = detail::read_input_multislice<double>(input);
    input_multislice.system_conf = system_conf;

    // Initialise the vectors
    mt::Vector< thrust::complex<double>, mt::e_host > psi_in(input_multislice.grid_2d.nxy());
    mt::Vector< thrust::complex<double>, mt::e_host > psi_out(input_multislice.grid_2d.nxy());
    psi_in.assign(psi_in.size(), 1.0);
      
    // Open a stream
    mt::Stream<mt::e_host> stream(system_conf.nstream);

    // Compute CTF
    switch (input_multislice.illumination_model) {
      case mt::eIM_Coherent:
				mt::apply_CTF(
            stream, 
            input_multislice.grid_2d, 
            input_multislice.obj_lens, 
            0, 
            0, 
            psi_in, 
            psi_out);
        break;
      case mt::eIM_Partial_Coherent:
        mt::apply_PCTF(
            stream, 
            input_multislice.grid_2d, 
            input_multislice.obj_lens, 
            psi_in, 
            psi_out);
        break;
      case mt::eIM_Partial_Coherent_Higher_Order:
        mt::apply_PCTF_HO(
            stream, 
            input_multislice.grid_2d, 
            input_multislice.obj_lens, 
            psi_in, 
            psi_out);
          break;
      default:
        break;
    }
    
    // Add phase shift
    thrust::transform(
          psi_out.begin(), 
          psi_out.end(), 
          psi_out.begin(), 
          mt::functor::scale<thrust::complex<double> >(
            exp(thrust::complex<double>(0, input_multislice.phase_shift))));

    // Syncronize stream
    stream.synchronize();

    // Return the image
    MULTEM_ASSERT(input_multislice.grid_2d.nx >= 0);
    MULTEM_ASSERT(input_multislice.grid_2d.ny >= 0);
    return Image< std::complex<double> >(
        psi_out.data(), 
        { 
          (std::size_t)input_multislice.grid_2d.nx,
          (std::size_t)input_multislice.grid_2d.ny 
        });
  }

  std::vector< std::pair<double, double> > compute_V_params(
      std::string potential_type, 
      std::size_t Z, 
      int charge) { 
    MULTEM_ASSERT(Z > 0);

    auto potential_type_enum = detail::from_string<mt::ePotential_Type>(potential_type);

    mt::Atom_Type<double, mt::e_host> atom_type;
    mt::Atomic_Data atomic_data(potential_type_enum);
    atomic_data.To_atom_type_CPU(Z, mt::c_Vrl, mt::c_nR, 0.0, atom_type);

    MULTEM_ASSERT(atom_type.coef.size() > 0);
    MULTEM_ASSERT(atom_type.coef[0].Vr.cl.size() == atom_type.coef[0].Vr.cnl.size());
    std::vector< std::pair<double, double> > result(atom_type.coef[0].Vr.cl.size());
    for (auto i = 0; i < result.size(); ++i) {
      result[i].first = atom_type.coef[0].Vr.cl[i];
      result[i].second = atom_type.coef[0].Vr.cnl[i];
    }
    return result;
    /* std::vector<double> a(atom_type.c_Vr[0].cl.begin(), atom_type.c_Vr[0].cl.end()); */
    /* std::vector<double> b(atom_type.c_Vr[0].cln.begin(), atom_type.c_Vr[0].cln.end()); */
    /* MULTEM_ASSERT(a.size() == b.size()); */
    /* std::vector< std::pair<double, double> > result(a.size()); */
    /* for (auto i = 0; i < result.size(); ++i) { */
    /*   result[i].first = a[i]; */
    /*   result[i].second = b[i]; */
    /* } */
    /* return result; */
  }

  bool is_gpu_available() {
    return mt::is_gpu_available();
  }

  double band_limit_spacing(double x_pixel_size, double y_pixel_size) {
    return detail::band_limit_spacing(x_pixel_size, y_pixel_size);
  }

  int number_of_gpu_available() {
    return mt::number_of_gpu_available();
  }
  
  double mrad_to_sigma(double E0, double theta) {
    return mt::rad_2_sigma(E0, theta * mt::c_mrad_2_rad);
  }

  double iehwgd_to_sigma(double value) {
    return mt::iehwgd_2_sigma(value);
  }
  
  double hwhm_to_sigma(double value) {
    return mt::hwhm_2_sigma(value);
  }

  std::vector<Atom> crystal_by_layers(const CrystalParameters &params) {
    
    // Get the layer data 
    mt::Vector<mt::Atom_Data<double>, mt::e_host> layers(params.layers.size());
    for(auto i = 0; i < params.layers.size(); ++i) {
      layers[i].resize(params.layers[i].size());
      for (auto j = 0; j < params.layers[i].size(); ++j) {
        layers[i].Z[j] = params.layers[i][j].element;
        layers[i].x[j] = params.layers[i][j].x;
        layers[i].y[j] = params.layers[i][j].y;
        layers[i].z[j] = params.layers[i][j].z;
        layers[i].sigma[j] = params.layers[i][j].sigma;
        layers[i].occ[j] = params.layers[i][j].occupancy;
        layers[i].region[j] = abs(params.layers[i][j].region);
        layers[i].charge[j] = params.layers[i][j].charge;
      }
    }

    // Get the atoms from the crystal specification
    mt::Crystal_Spec<double> crystal_spec;
    mt::Atom_Data<double> atoms;
    crystal_spec(
      params.na, 
      params.nb, 
      params.nc, 
      params.a, 
      params.b, 
      params.c, 
      layers, 
      atoms);

    // Copy to vector for output
    std::vector<Atom> result(atoms.size());
    for (auto i = 0; i < atoms.size(); ++i) {
      result[i].element = atoms.Z[i];
      result[i].x = atoms.x[i];
      result[i].y = atoms.y[i];
      result[i].z = atoms.z[i];
      result[i].sigma = atoms.sigma[i];
      result[i].occupancy = atoms.occ[i];
      result[i].region = atoms.region[i];
      result[i].charge = atoms.charge[i];
    }
    return result;
  }


  /**
   * Test to ensure that ice potential approximation gives correct results
   */
  template <typename FloatType, mt::eDevice DeviceType>
  void test_ice_potential_approximation_internal() {

    // Create the grid
    mt::Grid_2d<FloatType> grid_2d(1000,1000);

    // Create the FFT
    mt::FFT<FloatType, DeviceType> fft_2d;
    fft_2d.create_plan_2d(grid_2d.ny, grid_2d.nx, 1);

    // Create the masker
    Masker masker(grid_2d.nx, grid_2d.ny, 1.0);
    masker.set_cube({0,0,0}, 1000);

    // Create the random generator
    thrust::default_random_engine gen;

    // Create the ice potential
    multem::detail::IcePotentialApproximation<FloatType, DeviceType> potential_function;
    potential_function.set_fft_2d(&fft_2d);
    potential_function.set_grid(grid_2d);
    potential_function.set_random_engine(gen);
    potential_function.set_masker(masker);
    MULTEM_ASSERT(
        std::abs(
          potential_function.compute_number_density_of_water(1.0) -0.033428
        ) < 1e-6
    );

    // Compute
    mt::Vector<FloatType, DeviceType> V0_dev(grid_2d.nx*grid_2d.ny);
    potential_function(0, 5, V0_dev);
    mt::Vector<FloatType, mt::e_host> V0(V0_dev.begin(), V0_dev.end());

    // The CPU based function to see that we get the same result
    /* mt::Vector<FloatType, mt::e_host> V1; */
    /* { */

    /*   // The size of the slice */
    /*   double z_0 = 0; */
    /*   double z_e = 5; */

    /*   // Compute the mask */
    /*   mt::Vector<bool, mt::e_host> mask; */
    /*   mask.resize(V0.size()); */
    /*   masker.compute(z_0, z_e, mask.begin()); */
      
    /*   // The parameters to use */
    /*   double alpha_a1 = 3.61556763; */
    /*   double alpha_a2 = 23.22955402; */
    /*   double alpha_m1 = 5.48214868; */
    /*   double alpha_m2 = 11.81498691; */
    /*   double alpha_s1 = 2.27209584; */
    /*   double alpha_s2 = 3.64439385; */
    /*   double theta_a1 = 41.7597107; */
    /*   double theta_a2 = 1077.04791; */
    /*   double theta_m1 = 0.604256237; */
    /*   double theta_m2 = -10.0000000; */
    /*   double theta_s1 = 1.65486734; */
    /*   double theta_s2 = 35.4955295; */
    /*   double a1 = 1560.27916; */
    /*   double a2 = 4.41780420; */
    /*   double a3 = 693.960558; */
    /*   double m1 = 0.254522845; */
    /*   double m2 = 10.7305321; */
    /*   double m3 = 0.308002600; */
    /*   double s1 = 0.213959063; */
    /*   double s2 = 0.231840410; */
    /*   double s3 = 0.662902509; */

    /*   // The slice thickness */
    /*   double t = z_e - z_0; */
    /*   double alpha = */ 
    /*     alpha_a1 * std::exp(-0.5*(t-alpha_m1)*(t-alpha_m1)/(alpha_s1*alpha_s1)) + */ 
    /*     alpha_a2 * std::exp(-0.5*(t-alpha_m2)*(t-alpha_m2)/(alpha_s2*alpha_s2)); */ 
    /*   double theta = */ 
    /*     theta_a1 * std::exp(-0.5*(t-theta_m1)*(t-theta_m1)/(theta_s1*theta_s1)) + */ 
    /*     theta_a2 * std::exp(-0.5*(t-theta_m2)*(t-theta_m2)/(theta_s2*theta_s2)); */ 

    /*   // Compute the Gamma random field */
    /*   mt::Vector<FloatType, DeviceType> random_field; */
    /*   random_field.resize(V0.size()); */
        
    /*   // The size of the data */
    /*   std::size_t size = random_field.size(); */

    /*   // Create a uniform distribution random number generator */
    /*   thrust::uniform_real_distribution<double> uniform(0, 2*M_PI); */

    /*   mt::Vector<complex<FloatType>, mt::e_host> fft_data; */
    /*   mt::Vector<complex<FloatType>, DeviceType> fft_data_dev; */
    /*   fft_data.resize(size); */
    /*   fft_data_dev.resize(size); */
    /*   std::size_t xsize = grid_2d.nx; */
    /*   std::size_t ysize = grid_2d.ny; */
    /*   for (std::size_t j = 0; j < ysize; ++j) { */
    /*     for (std::size_t i = 0; i < xsize; ++i) { */
    /*       double xd = (i-xsize/2.0)/(xsize/2.0); */
    /*       double yd = (j-ysize/2.0)/(ysize/2.0); */
    /*       double r = std::sqrt(xd*xd + yd*yd); */
    /*       double power = */ 
    /*         a1 * std::exp(-0.5*(r - m1)*(r - m1) / (s1*s1)) + */ 
    /*         a2 * std::exp(-0.5*(r - m2)*(r - m2) / (s2*s2)) + */ 
    /*         a3 * std::exp(-0.5*(r - m3)*(r - m3) / (s3*s3)); */ 
    /*       double amplitude = std::sqrt(power); */
    /*       double phase = uniform(gen); */
    /*       fft_data[i+j*xsize] = amplitude * std::exp(std::complex<double>(0, phase)); */ 
    /*     } */
    /*   } */
    /*   fft_data_dev.assign(fft_data.begin(), fft_data.end()); */
    /*   mt::fft2_shift(grid_2d, fft_data_dev); */
    /*   fft_data_dev[0] = 0; */
    /*   fft_2d.inverse(fft_data_dev); */
      
    /*   mt::assign_real(fft_data_dev, random_field); */
      
    /*   mt::Vector<FloatType, mt::e_host> random_field_host; */
    /*   random_field_host.resize(size); */
    /*   random_field_host.assign(random_field.begin(), random_field.end()); */

    /*   double mean = 0; */
    /*   /1* for(auto x : random_field_host) { *1/ */
    /*   /1*   mean += x; *1/ */
    /*   /1* } *1/ */
    /*   /1* mean /= random_field_host.size(); *1/ */
    /*   double sdev = 0; */
    /*   for (auto x : random_field_host) { */
    /*     sdev += std::pow((x - mean), 2); */
    /*   } */
    /*   sdev = std::sqrt(sdev / random_field_host.size()); */
    /*   for (auto &x : random_field_host) { */
    /*     x = (x - mean) / sdev; */
    /*   } */

    /*   std::vector<double> gx(1000); */
    /*   std::vector<double> gy(1000); */
    /*   for (std::size_t i = 0; i < gx.size(); ++i) { */
    /*     gx[i] = (double)i / (double)gx.size(); */
    /*     /1* gy[i] = boost::math::gamma_p_inv(alpha, gx[i]) * theta; *1/ */
    /*   } */

    /*   for (std::size_t j = 0; j < ysize; ++j) { */
    /*     for (std::size_t i = 0; i < xsize; ++i) { */
    /*       auto &x = random_field_host[i+j*xsize]; */
    /*       if (mask[i+j*xsize]) { */
    /*         auto c = 0.5 * (1 + std::erf(x/std::sqrt(2))); */

    /*         double g = 0; */
    /*         int i = (std::size_t)std::floor(c * gx.size()); */
    /*         if (i < 0) { */
    /*           i = 0; */
    /*         } else if (i >= gx.size()-1) { */
    /*           i = gx.size()-2; */
    /*         } */
    /*         auto gx0 = gx[i]; */
    /*         auto gx1 = gx[i+1]; */
    /*         auto gy0 = gy[i]; */
    /*         auto gy1 = gy[i+1]; */
    /*         g = gy0 + (gy1 - gy0) / (gx1 - gx0) * (c - gx0); */

    /*         x = g; */
    /*       } else { */
    /*         x = 0; */
    /*       } */
    /*     } */
    /*   } */

    /*   mt::fft2_shift(grid_2d, random_field_host); */
    /*   V1.assign(random_field_host.begin(), random_field_host.end()); */
    /* } */

    /* for (std::size_t i = 0; i < V1.size(); ++i) { */
    /*   MULTEM_ASSERT(std::abs((V0[i] - V1[i])) < 1e-3); */
    /* } */

    // Cleanup
    fft_2d.cleanup();

    // If there was an error then throw an exception
    if (DeviceType == mt::e_device) {
      auto err = cudaGetLastError();
      if (err != cudaSuccess) {
        throw multem::Error(__FILE__, __LINE__, cudaGetErrorString(err));
      }
    }
  }

  /**
   * Run tests for all devices and precisions
   */
  void test_ice_potential_approximation() {
    test_ice_potential_approximation_internal<float, mt::e_host>();
    test_ice_potential_approximation_internal<double, mt::e_host>();
    /* test_ice_potential_approximation_internal<float, mt::e_device>(); */
    /* test_ice_potential_approximation_internal<double, mt::e_device>(); */
  }

	/**
   * Run tests for the cuboid masker
   */
	void test_cuboid_masker() {
    
    // Setup the masker
    Masker masker;
    masker.set_image_size(100, 200);
    masker.set_pixel_size(0.5);
    masker.set_cuboid({ 10, 11, 12 }, { 20, 30, 40 });
    masker.set_translation({4, 5, 6});
    masker.set_rotation({10, 11, 12}, { 0, M_PI / 2.0, 0 });

    // Check stuff
    MULTEM_ASSERT(masker.xsize() == 100);
    MULTEM_ASSERT(masker.ysize() == 200);
    MULTEM_ASSERT(masker.image_size() == 200*100);
    MULTEM_ASSERT(masker.pixel_size() == 0.5);
    MULTEM_ASSERT(masker.shape() == Masker::Cuboid);
    MULTEM_ASSERT(std::abs(masker.xmin() - (10 + 4)) < 1e-5);
    MULTEM_ASSERT(std::abs(masker.ymin() - (11 + 5)) < 1e-5);
    MULTEM_ASSERT(std::abs(masker.zmin() - (12 - 20 + 6)) < 1e-5);
    MULTEM_ASSERT(std::abs(masker.xmax() - (10 + 40 + 4)) < 1e-5);
    MULTEM_ASSERT(std::abs(masker.ymax() - (11 + 5 + 30)) < 1e-5);
    MULTEM_ASSERT(std::abs(masker.zmax() - (12 + 6)) < 1e-5);
    MULTEM_ASSERT(masker.rotation_origin()[0] == 10);
    MULTEM_ASSERT(masker.rotation_origin()[1] == 11);
    MULTEM_ASSERT(masker.rotation_origin()[2] == 12);
    MULTEM_ASSERT(std::abs(masker.rotation_angle() - M_PI/2) < 1e-5);
    MULTEM_ASSERT(masker.translation()[0] == 4);
    MULTEM_ASSERT(masker.translation()[1] == 5);
    MULTEM_ASSERT(masker.translation()[2] == 6);

    // Compute the mask
    std::vector<bool> mask(masker.image_size());
    for (std::size_t z = 0; z < 100; ++z) {
      masker.compute(z, z+1, mask.begin());
      for (std::size_t j = 0; j < 200; ++j) {
        for (std::size_t i = 0; i < 100; ++i) {
          double x = i * masker.pixel_size();
          double y = j * masker.pixel_size();
          bool value = false;
          if (x >= masker.xmin() && x < masker.xmax() && 
              y >= masker.ymin() && y < masker.ymax() &&
              z >= masker.zmin() && z < masker.zmax()) {
            value = true;
          }
          /* if (mask[j+i*masker.ysize()] != value) { */
          /*   std::cout << x << ", " << y << ", " << z << ", " << value << std::endl; */
          /* } */
          //MULTEM_ASSERT(mask[j+i*masker.ysize()] == value);
        }
      }
    }
	}

	/**
   * Run tests for the cylinder masker
   */
  void test_cylinder_masker() {

    // Setup the masker
    Masker masker;
    masker.set_image_size(100, 200);
    masker.set_pixel_size(0.5);
    masker.set_cylinder({ 10, 11, 12 }, { 0, 1, 0 }, 50, {12}, {0}, {0});
    masker.set_translation({4, 5, 6});
    masker.set_rotation({10, 11, 12}, { 0, M_PI / 2.0, 0 });

    // Check stuff
    MULTEM_ASSERT(masker.xsize() == 100);
    MULTEM_ASSERT(masker.ysize() == 200);
    MULTEM_ASSERT(masker.image_size() == 200*100);
    MULTEM_ASSERT(masker.pixel_size() == 0.5);
    MULTEM_ASSERT(masker.shape() == Masker::Cylinder);
    MULTEM_ASSERT(std::abs(masker.xmin() - (10 + 4)) < 1e-5);
    MULTEM_ASSERT(std::abs(masker.ymin() - (11 + 5)) < 1e-5);
    /* MULTEM_ASSERT(std::abs(masker.zmin() - (12 + 6)) < 1e-5); */
    MULTEM_ASSERT(std::abs(masker.xmax() - (10 + 4)) < 1e-5);
    MULTEM_ASSERT(std::abs(masker.ymax() - (11 + 5 + 50)) < 1e-5);
    /* MULTEM_ASSERT(std::abs(masker.zmax() - (12 + 6)) < 1e-5); */
    MULTEM_ASSERT(masker.rotation_origin()[0] == 10);
    MULTEM_ASSERT(masker.rotation_origin()[1] == 11);
    MULTEM_ASSERT(masker.rotation_origin()[2] == 12);
    MULTEM_ASSERT(std::abs(masker.rotation_angle() - M_PI/2) < 1e-5);
    MULTEM_ASSERT(masker.translation()[0] == 4);
    MULTEM_ASSERT(masker.translation()[1] == 5);
    MULTEM_ASSERT(masker.translation()[2] == 6);

    // Compute the mask
    std::vector<bool> mask(masker.image_size());
    for (std::size_t z = 0; z < 100; ++z) {
      masker.compute(z, z+1, mask.begin());
      for (std::size_t j = 0; j < 200; ++j) {
        for (std::size_t i = 0; i < 100; ++i) {
          double x = (i + 0.5) * masker.pixel_size();
          double y = (j + 0.5) * masker.pixel_size();
          double xc = 10 + 4;
          double zc = 12 + 6;
          double d = std::sqrt((x-xc)*(x-xc)+(z+0.5-zc)*(z+0.5-zc));
          double radius = 12;
          bool value = false;
          if (y >= masker.ymin() && y < masker.ymax() && d < radius) {
            value = true;
          }
          //MULTEM_ASSERT(mask[j+i*masker.ysize()] == value);
        }
      }
    }

  }

  /**
   * Run the ice parameter tests
   */
  void test_ice_parameters() {
    const double TINY = 1e-7;
    IceParameters ice_parameters; 
    Masker masker;

    masker.set_ice_parameters(ice_parameters);
    MULTEM_ASSERT(std::abs(masker.ice_parameters().m1 - 0) < TINY);
    MULTEM_ASSERT(std::abs(masker.ice_parameters().m2 - 1.0/2.88) < TINY);
    MULTEM_ASSERT(std::abs(masker.ice_parameters().s1 - 0.731) < TINY);
    MULTEM_ASSERT(std::abs(masker.ice_parameters().s2 - 0.081) < TINY);
    MULTEM_ASSERT(std::abs(masker.ice_parameters().a1 - 0.199) < TINY);
    MULTEM_ASSERT(std::abs(masker.ice_parameters().a2 - 0.801) < TINY);
    MULTEM_ASSERT(std::abs(masker.ice_parameters().density - 0.91) < TINY);

    ice_parameters.m1 = 10;
    ice_parameters.m2 = 20;
    ice_parameters.s1 = 30;
    ice_parameters.s2 = 40;
    ice_parameters.a1 = 50;
    ice_parameters.a2 = 60;
    ice_parameters.density = 1;
    
    masker.set_ice_parameters(ice_parameters);
    MULTEM_ASSERT(std::abs(masker.ice_parameters().m1 - 10) < TINY);
    MULTEM_ASSERT(std::abs(masker.ice_parameters().m2 - 20) < TINY);
    MULTEM_ASSERT(std::abs(masker.ice_parameters().s1 - 30) < TINY);
    MULTEM_ASSERT(std::abs(masker.ice_parameters().s2 - 40) < TINY);
    MULTEM_ASSERT(std::abs(masker.ice_parameters().a1 - 50) < TINY);
    MULTEM_ASSERT(std::abs(masker.ice_parameters().a2 - 60) < TINY);
    MULTEM_ASSERT(std::abs(masker.ice_parameters().density - 1) < TINY);
  }

	/**
   * Run tests for the masker
   */
	void test_masker() {
		test_cuboid_masker();
		test_cylinder_masker();
    test_ice_parameters();
	}


  // ── Phase E: X-ray multislice simulation ──────────────────────────────

  XrayExitWave simulate_xray(
      const MultiRegionMasker &masker,
      const std::vector<XrayRegionInfo> &regions,
      double wavelength,
      double pixel_size,
      double slice_thickness,
      std::size_t nx,
      std::size_t ny) {

    MULTEM_ASSERT(regions.size() == masker.num_regions());
    MULTEM_ASSERT(wavelength > 0);
    MULTEM_ASSERT(pixel_size > 0);
    MULTEM_ASSERT(slice_thickness > 0);
    MULTEM_ASSERT(nx > 0 && ny > 0);

    const double c_Pi = 3.141592653589793238463;
    const double k = 2.0 * c_Pi / wavelength;  // wavenumber
    const std::size_t npix = nx * ny;

    // Determine z-range from masker geometry
    double z_min = masker.zmin();
    double z_max = masker.zmax();
    MULTEM_ASSERT(z_max > z_min);

    // Number of slices
    double specimen_thickness = z_max - z_min;
    int n_slices = std::max(1, (int)std::ceil(specimen_thickness / slice_thickness));
    double dz = specimen_thickness / n_slices;

    // Build Fresnel propagator in Fourier space: exp(i*pi*lambda*dz*f^2)
    // Frequency grid: fx[i] = i/N/pixel_size for i < N/2, (i-N)/N/pixel_size for i >= N/2
    std::vector< std::complex<double> > propagator(npix);
    for (std::size_t iy = 0; iy < ny; ++iy) {
      double fy = (iy <= ny/2) ? (double)iy / (ny * pixel_size)
                                : ((double)iy - (double)ny) / (ny * pixel_size);
      for (std::size_t ix = 0; ix < nx; ++ix) {
        double fx = (ix <= nx/2) ? (double)ix / (nx * pixel_size)
                                  : ((double)ix - (double)nx) / (nx * pixel_size);
        double f2 = fx*fx + fy*fy;
        double phase = c_Pi * wavelength * dz * f2;
        propagator[iy * nx + ix] = std::complex<double>(cos(phase), sin(phase));
      }
    }

    // Allocate per-region mask buffers (host)
    // NOTE: use uint8_t, not bool — std::vector<bool> is bitpacked with
    // _Bit_iterator that is incompatible with thrust::transform in CUDA 12.
    std::vector< std::vector<uint8_t> > region_masks(regions.size(),
                                                      std::vector<uint8_t>(npix, 0));
    std::vector<uint8_t> scratch(npix, 0);

    // Initialise wave function: psi = 1 (plane wave)
    std::vector< std::complex<double> > psi(npix, std::complex<double>(1.0, 0.0));
    std::vector< std::complex<double> > psi_ft(npix);
    std::vector< std::complex<double> > trans(npix);

    // Allocate per-pixel delta/beta arrays
    std::vector<double> delta_map(npix);
    std::vector<double> beta_map(npix);

    // Create FFTW plans for 2D forward and inverse transforms.
    // FFTW_ESTIMATE avoids overwriting input arrays during planning.
    // Note: fftw_plan_dft_2d takes (rows, cols) = (ny, nx).
    fftw_plan plan_fwd = fftw_plan_dft_2d(
        (int)ny, (int)nx,
        reinterpret_cast<fftw_complex*>(psi.data()),
        reinterpret_cast<fftw_complex*>(psi_ft.data()),
        FFTW_FORWARD, FFTW_ESTIMATE);
    fftw_plan plan_inv = fftw_plan_dft_2d(
        (int)ny, (int)nx,
        reinterpret_cast<fftw_complex*>(psi_ft.data()),
        reinterpret_cast<fftw_complex*>(psi.data()),
        FFTW_BACKWARD, FFTW_ESTIMATE);

    // ─── Multislice loop ────────────────────────────────────────────
    for (int islice = 0; islice < n_slices; ++islice) {
      double z_0 = z_min + islice * dz;
      double z_e = z_0 + dz;

      // Compute masks for all regions in this slice
      for (std::size_t r = 0; r < regions.size(); ++r) {
        // Use masker_compute_mask for the region's shape
        const auto &region = masker.region(r);
        detail::masker_compute_mask(region.shape, z_0, z_e,
                                    region_masks[r].begin());

        // Apply boolean subtraction
        for (int sub_idx : region.subtract_regions) {
          MULTEM_ASSERT(sub_idx >= 0 && sub_idx < (int)masker.num_regions());
          const auto &sub_region = masker.region(sub_idx);
          detail::masker_compute_mask(sub_region.shape, z_0, z_e,
                                      scratch.begin());
          for (std::size_t p = 0; p < npix; ++p) {
            if (scratch[p]) region_masks[r][p] = 0;
          }
        }
      }

      // Build per-pixel (delta, beta) from region masks
      // Pixels not covered by any region: vacuum (delta=0, beta=0)
      std::fill(delta_map.begin(), delta_map.end(), 0.0);
      std::fill(beta_map.begin(),  beta_map.end(),  0.0);

      // Last-region-wins: later regions override earlier ones
      for (std::size_t r = 0; r < regions.size(); ++r) {
        double d = regions[r].delta;
        double b = regions[r].beta;
        for (std::size_t p = 0; p < npix; ++p) {
          if (region_masks[r][p]) {
            delta_map[p] = d;
            beta_map[p] = b;
          }
        }
      }

      // Build complex transmission function:
      // t(x,y) = exp[-(2*pi/lambda) * (beta + i*delta) * dz]
      for (std::size_t p = 0; p < npix; ++p) {
        double amp_arg = -k * beta_map[p] * dz;
        double phase_arg = -k * delta_map[p] * dz;
        double amp = exp(amp_arg);
        trans[p] = std::complex<double>(amp * cos(phase_arg),
                                         amp * sin(phase_arg));
      }

      // Apply transmission: psi *= trans
      for (std::size_t p = 0; p < npix; ++p) {
        psi[p] *= trans[p];
      }

      // Fresnel propagation: psi = IFFT(FFT(psi) * propagator)
      // Using FFTW for O(N log N) performance.
      // Forward FFT: psi -> psi_ft
      fftw_execute_dft(plan_fwd,
          reinterpret_cast<fftw_complex*>(psi.data()),
          reinterpret_cast<fftw_complex*>(psi_ft.data()));

      // Multiply by propagator in Fourier space
      for (std::size_t p = 0; p < npix; ++p) {
        psi_ft[p] *= propagator[p];
      }

      // Inverse FFT: psi_ft -> psi
      fftw_execute_dft(plan_inv,
          reinterpret_cast<fftw_complex*>(psi_ft.data()),
          reinterpret_cast<fftw_complex*>(psi.data()));

      // FFTW inverse is unnormalized; divide by N
      double inv_npix = 1.0 / (double)npix;
      for (std::size_t p = 0; p < npix; ++p) {
        psi[p] *= inv_npix;
      }
    }

    // Destroy FFTW plans
    fftw_destroy_plan(plan_fwd);
    fftw_destroy_plan(plan_inv);

    // Build result
    XrayExitWave result;
    result.nx = nx;
    result.ny = ny;
    result.psi = std::move(psi);
    return result;
  }

}

