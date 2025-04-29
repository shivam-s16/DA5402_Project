/** @type {import('next').NextConfig} */
const nextConfig = {
  eslint: {
    ignoreDuringBuilds: true,
  },
  typescript: {
    ignoreBuildErrors: true,
  },
  images: {
    remotePatterns: [
      {
        protocol: 'https',
        hostname: 'news.google.com',
        pathname: '/api/attachments/**',
      },
      // Add more patterns for other domains if needed:
      // {
      //   protocol: 'https',
      //   hostname: 'anotherdomain.com',
      //   pathname: '/path/**',
      // },
    ],
  },
}

export default nextConfig
