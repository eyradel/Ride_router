"use client";

import Link from "next/link";
import dynamic from "next/dynamic";
import { useState, useEffect } from "react";
import { useForm } from "react-hook-form";
import { zodResolver } from "@hookform/resolvers/zod";
import { z } from "zod";
import { useRouter } from "next/navigation";
import { toast } from "react-hot-toast";
import { auth } from "@/lib/api";
import { FiLock, FiMail, FiUser, FiPhone, FiGlobe, FiBriefcase } from "react-icons/fi";
import type { IconType } from "react-icons";
import { CSSTransition, TransitionGroup } from "react-transition-group";

const MapContainer = dynamic(
  () => import("react-leaflet").then(mod => mod.MapContainer),
  { ssr: false }
);
const TileLayer = dynamic(
  () => import("react-leaflet").then(mod => mod.TileLayer),
  { ssr: false }
);
const Marker = dynamic(
  () => import("react-leaflet").then(mod => mod.Marker),
  { ssr: false }
);
const Popup = dynamic(
  () => import("react-leaflet").then(mod => mod.Popup),
  { ssr: false }
);

import "leaflet/dist/leaflet.css";

const registerSchema = z.object({
  email: z.string().email("Invalid email address"),
  username: z.string().min(2, "Username is required"),
  full_name: z.string().min(2, "Full name is required"),
  mobile_number: z.string().min(5, "Mobile number is required"),
  country: z.string().min(2, "Country is required"),
  company: z.string().min(2, "Company is required"),
  role: z.enum(["staff", "driver", "admin"]),
  password: z.string().min(6, "Password must be at least 6 characters"),
  confirmPassword: z.string().min(6, "Please confirm your password"),
}).refine((data) => data.password === data.confirmPassword, {
  message: "Passwords do not match",
  path: ["confirmPassword"],
});

type RegisterData = z.infer<typeof registerSchema>;

const steps = [
  "Account",
  "Personal",
  "Company",
  "Password",
];

export default function Register() {
  const router = useRouter();
  const [loading, setLoading] = useState(false);
  const [position, setPosition] = useState<[number, number]>([6.5244, 3.3792]);
  const [markerIcon, setMarkerIcon] = useState<any>(null);
  const [step, setStep] = useState(0);

  const {
    register,
    handleSubmit,
    formState: { errors },
    trigger,
  } = useForm<RegisterData>({
    resolver: zodResolver(registerSchema),
    defaultValues: { role: "staff" },
    mode: "onTouched",
  });

  useEffect(() => {
    if (typeof window !== "undefined" && navigator.geolocation) {
      navigator.geolocation.getCurrentPosition(
        (pos) => {
          setPosition([pos.coords.latitude, pos.coords.longitude]);
        },
        () => {},
        { enableHighAccuracy: true }
      );
    }
    import("leaflet").then(L => {
      setMarkerIcon(
        new L.Icon({
          iconUrl: "https://unpkg.com/leaflet@1.9.3/dist/images/marker-icon.png",
          iconSize: [32, 48],
          iconAnchor: [16, 48],
          popupAnchor: [0, -48],
        })
      );
    });
  }, []);

  const onSubmit = async (data: RegisterData) => {
    try {
      setLoading(true);
      const response = await auth.register(data);
      console.log("Registration response:", response); // Debug log
      toast.success("Registration successful! Please check your email and phone for verification code.");
      
      // Force a hard navigation to ensure the redirect works
      window.location.href = "/verify";
    } catch (error: any) {
      console.error("Registration error:", error); // Debug log
      toast.error(error.message || "Registration failed");
    } finally {
      setLoading(false);
    }
  };

  // Step navigation
  const nextStep = async () => {
    let valid = false;
    if (step === 0) valid = await trigger(["email", "username"]);
    if (step === 1) valid = await trigger(["full_name", "mobile_number"]);
    if (step === 2) valid = await trigger(["country", "company", "role"]);
    if (step === 3) valid = await trigger(["password", "confirmPassword"]);
    if (valid) setStep((s) => Math.min(s + 1, steps.length - 1));
  };
  const prevStep = () => setStep((s) => Math.max(s - 1, 0));

  // Step content
  const renderStep = () => {
    switch (step) {
      case 0:
        return (
          <>
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">Email Address</label>
              <div className="relative">
                <span className="absolute inset-y-0 left-0 flex items-center pl-3">
                  <FiMail className="h-5 w-5 text-gray-400" />
                </span>
                <input
                  type="email"
                  {...register("email")}
                  className="pl-10 block w-full rounded-xl border border-gray-200 shadow-sm focus:border-orange-500 focus:ring-orange-500 bg-white/80 placeholder-gray-400"
                  placeholder="Enter your email"
                />
              </div>
              {errors.email && <p className="mt-1 text-xs text-red-600">{errors.email.message}</p>}
            </div>
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">Username</label>
              <div className="relative">
                <span className="absolute inset-y-0 left-0 flex items-center pl-3">
                  <FiUser className="h-5 w-5 text-gray-400" />
                </span>
                <input
                  type="text"
                  {...register("username")}
                  className="pl-10 block w-full rounded-xl border border-gray-200 shadow-sm focus:border-orange-500 focus:ring-orange-500 bg-white/80 placeholder-gray-400"
                  placeholder="Choose a username"
                />
              </div>
              {errors.username && <p className="mt-1 text-xs text-red-600">{errors.username.message}</p>}
            </div>
          </>
        );
      case 1:
        return (
          <>
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">Full Name</label>
              <div className="relative">
                <span className="absolute inset-y-0 left-0 flex items-center pl-3">
                  <FiUser className="h-5 w-5 text-gray-400" />
                </span>
                <input
                  type="text"
                  {...register("full_name")}
                  className="pl-10 block w-full rounded-xl border border-gray-200 shadow-sm focus:border-orange-500 focus:ring-orange-500 bg-white/80 placeholder-gray-400"
                  placeholder="Enter your full name"
                />
              </div>
              {errors.full_name && <p className="mt-1 text-xs text-red-600">{errors.full_name.message}</p>}
            </div>
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">Mobile Number</label>
              <div className="relative">
                <span className="absolute inset-y-0 left-0 flex items-center pl-3">
                  <FiPhone className="h-5 w-5 text-gray-400" />
                </span>
                <input
                  type="text"
                  {...register("mobile_number")}
                  className="pl-10 block w-full rounded-xl border border-gray-200 shadow-sm focus:border-orange-500 focus:ring-orange-500 bg-white/80 placeholder-gray-400"
                  placeholder="Enter your mobile number"
                />
              </div>
              {errors.mobile_number && <p className="mt-1 text-xs text-red-600">{errors.mobile_number.message}</p>}
            </div>
          </>
        );
      case 2:
        return (
          <>
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">Country</label>
              <div className="relative">
                <span className="absolute inset-y-0 left-0 flex items-center pl-3">
                  <FiGlobe className="h-5 w-5 text-gray-400" />
                </span>
                <input
                  type="text"
                  {...register("country")}
                  className="pl-10 block w-full rounded-xl border border-gray-200 shadow-sm focus:border-orange-500 focus:ring-orange-500 bg-white/80 placeholder-gray-400"
                  placeholder="Enter your country"
                />
              </div>
              {errors.country && <p className="mt-1 text-xs text-red-600">{errors.country.message}</p>}
            </div>
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">Company</label>
              <div className="relative">
                <span className="absolute inset-y-0 left-0 flex items-center pl-3">
                  <FiBriefcase className="h-5 w-5 text-gray-400" />
                </span>
                <input
                  type="text"
                  {...register("company")}
                  className="pl-10 block w-full rounded-xl border border-gray-200 shadow-sm focus:border-orange-500 focus:ring-orange-500 bg-white/80 placeholder-gray-400"
                  placeholder="Enter your company"
                />
              </div>
              {errors.company && <p className="mt-1 text-xs text-red-600">{errors.company.message}</p>}
            </div>
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">Role</label>
              <div className="relative">
                <select
                  {...register("role")}
                  className="block w-full rounded-xl border border-gray-200 shadow-sm focus:border-orange-500 focus:ring-orange-500 bg-white/80 pl-3 pr-10 text-gray-700"
                >
                  <option value="staff">Staff</option>
                  <option value="driver">Driver</option>
                  <option value="admin">Admin</option>
                </select>
              </div>
              {errors.role && <p className="mt-1 text-xs text-red-600">{errors.role.message}</p>}
            </div>
          </>
        );
      case 3:
        return (
          <>
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">Password</label>
              <div className="relative">
                <span className="absolute inset-y-0 left-0 flex items-center pl-3">
                  <FiLock className="h-5 w-5 text-gray-400" />
                </span>
                <input
                  type="password"
                  {...register("password")}
                  className="pl-10 block w-full rounded-xl border border-gray-200 shadow-sm focus:border-orange-500 focus:ring-orange-500 bg-white/80 placeholder-gray-400"
                  placeholder="Create a password"
                />
              </div>
              {errors.password && <p className="mt-1 text-xs text-red-600">{errors.password.message}</p>}
            </div>
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">Confirm Password</label>
              <div className="relative">
                <span className="absolute inset-y-0 left-0 flex items-center pl-3">
                  <FiLock className="h-5 w-5 text-gray-400" />
                </span>
                <input
                  type="password"
                  {...register("confirmPassword")}
                  className="pl-10 block w-full rounded-xl border border-gray-200 shadow-sm focus:border-orange-500 focus:ring-orange-500 bg-white/80 placeholder-gray-400"
                  placeholder="Confirm your password"
                />
              </div>
              {errors.confirmPassword && <p className="mt-1 text-xs text-red-600">{errors.confirmPassword.message}</p>}
            </div>
          </>
        );
      default:
        return null;
    }
  };

  return (
    <div className="flex min-h-screen">
      {/* Map on left for desktop */}
      <div className="hidden lg:block w-1/2 h-screen z-0">
        {markerIcon && (
          <MapContainer
            center={position}
            zoom={13}
            scrollWheelZoom={true}
            style={{ height: "100vh", width: "100%", transition: "opacity 0.5s" }}
            className="fade-in"
          >
            <TileLayer
              url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
              attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
            />
            <Marker position={position} icon={markerIcon}>
              <Popup>
                You are here
              </Popup>
            </Marker>
          </MapContainer>
        )}
      </div>
      {/* Form on right (full width on mobile/tablet) */}
      <div className="flex flex-col items-center justify-center w-full lg:w-1/2 min-h-screen bg-transparent px-2 sm:px-8">
        <form onSubmit={handleSubmit(onSubmit)} className="backdrop-blur-xl bg-white/80 border border-white/40 rounded-3xl shadow-2xl px-4 py-6 sm:px-8 sm:py-10 w-full max-w-lg animate-fade-in-up">
          <h2 className="text-xl sm:text-2xl font-bold text-gray-900 mb-2 text-center">Create your Ride Router account</h2>
          <div className="flex justify-center mb-6">
            {steps.map((s, i) => (
              <div key={s} className={`h-2 w-8 mx-1 rounded-full ${i <= step ? 'bg-orange-500' : 'bg-gray-200'}`}></div>
            ))}
          </div>
          <TransitionGroup>
            <CSSTransition key={step} classNames="slide" timeout={300}>
              <div className="grid grid-cols-1 gap-4 w-full">{renderStep()}</div>
            </CSSTransition>
          </TransitionGroup>
          <div className="flex justify-between mt-8 w-full">
            <button
              type="button"
              onClick={prevStep}
              disabled={step === 0}
              className="px-4 py-2 rounded-lg bg-gray-200 text-gray-700 font-medium disabled:opacity-50"
            >
              Back
            </button>
            {step < steps.length - 1 ? (
              <button
                type="button"
                onClick={nextStep}
                className="px-6 py-2 rounded-lg bg-orange-500 text-white font-semibold hover:bg-orange-600 transition"
              >
                Next
              </button>
            ) : (
              <button
                type="submit"
                disabled={loading}
                className="px-6 py-2 rounded-lg bg-gradient-to-r from-orange-500 to-orange-600 text-white font-semibold shadow-lg hover:from-orange-600 hover:to-orange-700 transition focus:outline-none focus:ring-2 focus:ring-orange-400 focus:ring-offset-2 disabled:opacity-60"
              >
                {loading ? (
                  <div className="flex items-center justify-center">
                    <div className="w-5 h-5 border-t-2 border-b-2 border-white rounded-full animate-spin mr-2"></div>
                    Signing up...
                  </div>
                ) : (
                  "Sign up"
                )}
              </button>
            )}
          </div>
          <div className="text-center mt-4 w-full">
            <span className="text-gray-500">Already have an account? </span>
            <Link href="/login" className="text-orange-600 hover:underline font-medium">
              Login
            </Link>
          </div>
        </form>
      </div>
      <style jsx global>{`
        .fade-in {
          opacity: 0;
          animation: fadeIn 1s forwards;
        }
        @keyframes fadeIn {
          to {
            opacity: 1;
          }
        }
        .animate-fade-in-up {
          opacity: 0;
          transform: translateY(40px);
          animation: fadeInUp 0.8s 0.2s forwards;
        }
        @keyframes fadeInUp {
          to {
            opacity: 1;
            transform: translateY(0);
          }
        }
        .slide-enter {
          opacity: 0;
          transform: translateX(100%);
        }
        .slide-enter-active {
          opacity: 1;
          transform: translateX(0);
          transition: all 300ms;
        }
        .slide-exit {
          opacity: 1;
          transform: translateX(0);
        }
        .slide-exit-active {
          opacity: 0;
          transform: translateX(-100%);
          transition: all 300ms;
        }
      `}</style>
    </div>
  );
}
