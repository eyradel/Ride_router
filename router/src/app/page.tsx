"use client";

import Link from "next/link";
import dynamic from "next/dynamic";
import { useEffect, useState } from "react";

// Dynamically import MapContainer and related components to avoid SSR issues
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

export default function Home() {
  const [position, setPosition] = useState<[number, number]>([6.5244, 3.3792]); // Default: Lagos, Nigeria
  const [markerIcon, setMarkerIcon] = useState<any>(null);

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
    // Dynamically import leaflet and set marker icon on client only
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

  return (
    <div className="fixed inset-0 w-screen h-screen overflow-hidden z-0">
      {/* Full-screen Map */}
      <div className="fixed inset-0 z-0">
        {markerIcon && (
          <MapContainer
            center={position}
            zoom={13}
            scrollWheelZoom={true}
            style={{ height: "100vh", width: "100vw", transition: "opacity 0.5s" }}
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

      {/* Top Bar */}
      <nav className="fixed top-0 left-0 w-full flex items-center justify-between px-6 py-4 z-20 bg-black/40 backdrop-blur-md">
        <div className="flex items-center space-x-2">
          <span className="text-2xl">🚕</span>
          <span className="text-xl font-bold text-white tracking-tight">Ride Router</span>
        </div>
        <div className="hidden md:flex space-x-6 text-white font-medium">
          <Link href="#" className="hover:text-orange-400 transition">Home</Link>
          <Link href="#" className="hover:text-orange-400 transition">About</Link>
          <Link href="#" className="hover:text-orange-400 transition">Contact</Link>
        </div>
      </nav>

      {/* Floating Action Card */}
      <div className="fixed left-1/2 bottom-10 transform -translate-x-1/2 z-20 w-full max-w-sm px-4">
        <div className="bg-white/95 rounded-2xl shadow-2xl px-6 py-8 flex flex-col items-center animate-fade-in-up">
          <h2 className="text-xl font-bold text-gray-900 mb-2 text-center">Where to?</h2>
          <input
            type="text"
            placeholder="Enter destination"
            className="w-full mb-4 px-4 py-3 rounded-xl border border-gray-200 focus:border-orange-500 focus:ring-orange-500 text-lg shadow-sm transition"
          />
          <div className="flex w-full space-x-3 mb-4">
            <Link href="/login" className="flex-1 bg-black text-white py-3 rounded-xl font-semibold text-lg shadow hover:bg-gray-900 transition text-center flex items-center justify-center">Login</Link>
            <Link href="/register" className="flex-1 bg-orange-500 text-white py-3 rounded-xl font-semibold text-lg shadow hover:bg-orange-600 transition text-center flex items-center justify-center">Register</Link>
          </div>
          <p className="text-gray-500 text-sm text-center">Book a cab, tricycle, or bike ride in seconds.</p>
        </div>
      </div>

      {/* Mobile nav menu icon (optional) */}
      {/* <div className="absolute top-4 right-4 z-30 md:hidden">
        <button className="text-white text-3xl">
          <FiMenu />
        </button>
      </div> */}

      {/* Fade-in animation styles */}
      <style jsx global>{`
        html, body, #__next {
          height: 100%;
          width: 100%;
          margin: 0;
          padding: 0;
          overflow: hidden;
        }
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
      `}</style>
    </div>
  );
}
