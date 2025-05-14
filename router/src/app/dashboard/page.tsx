"use client";

import { useState, useEffect } from "react";
import dynamic from "next/dynamic";
import { useRouter } from "next/navigation";
import { toast } from "react-hot-toast";
import { FiMapPin, FiUser, FiLogOut } from "react-icons/fi";
import { auth } from "@/lib/api";

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

export default function Dashboard() {
  const router = useRouter();
  const [position, setPosition] = useState<[number, number]>([6.5244, 3.3792]);
  const [markerIcon, setMarkerIcon] = useState<any>(null);
  const [user, setUser] = useState<any>(null);

  useEffect(() => {
    // Redirect to login if not authenticated
    const token = typeof window !== "undefined" ? localStorage.getItem("token") : null;
    if (!token) {
      router.push("/login");
      return;
    }
    // Get user's current position
    if (typeof window !== "undefined" && navigator.geolocation) {
      navigator.geolocation.getCurrentPosition(
        (pos) => {
          setPosition([pos.coords.latitude, pos.coords.longitude]);
        },
        () => {
          toast.error("Unable to get your location");
        },
        { enableHighAccuracy: true }
      );
    }

    // Initialize map marker icon
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

    // Get user profile
    const fetchUser = async () => {
      try {
        const userData = await auth.getProfile();
        setUser(userData);
      } catch (error) {
        console.error("Error fetching user profile:", error);
        toast.error("Failed to load user profile");
      }
    };

    fetchUser();
  }, []);

  const handleLogout = async () => {
    try {
      await auth.logout();
      toast.success("Logged out successfully");
      router.push("/login");
    } catch (error) {
      toast.error("Failed to logout");
    }
  };

  return (
    <div className="h-screen flex flex-col">
      {/* Top Navigation Bar */}
      <nav className="bg-white shadow-sm z-10">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex justify-between h-16">
            <div className="flex items-center">
              <span className="text-2xl">🚕</span>
              <span className="ml-2 text-xl font-bold text-gray-900">Ride Router</span>
            </div>
            <div className="flex items-center space-x-4">
              <div className="flex items-center text-gray-700">
                <FiUser className="h-5 w-5 mr-2" />
                <span>{user?.full_name || "Loading..."}</span>
              </div>
              <button
                onClick={handleLogout}
                className="flex items-center text-gray-700 hover:text-orange-600"
              >
                <FiLogOut className="h-5 w-5 mr-2" />
                Logout
              </button>
            </div>
          </div>
        </div>
      </nav>

      {/* Main Content */}
      <div className="flex-1 relative">
        {markerIcon && (
          <MapContainer
            center={position}
            zoom={13}
            scrollWheelZoom={true}
            style={{ height: "100%", width: "100%" }}
          >
            <TileLayer
              url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
              attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
            />
            <Marker position={position} icon={markerIcon}>
              <Popup>
                <div className="text-center">
                  <FiMapPin className="h-5 w-5 mx-auto mb-2 text-orange-500" />
                  <p className="font-medium">You are here</p>
                  <p className="text-sm text-gray-500">
                    {position[0].toFixed(4)}, {position[1].toFixed(4)}
                  </p>
                </div>
              </Popup>
            </Marker>
          </MapContainer>
        )}
      </div>
    </div>
  );
} 