/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,
  swcMinify: true,
  images: {
    domains: ['tile.openstreetmap.org'],
  },
};

module.exports = nextConfig; 