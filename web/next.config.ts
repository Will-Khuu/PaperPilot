import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  experimental: {
    // Match Streamlit's 180-second wait so paid answers can finish before the proxy gives up.
    proxyTimeout: 180_000,
    // Allow the API's 20 MB PDFs plus multipart overhead without truncating the upload.
    proxyClientMaxBodySize: "21mb",
  },
  async rewrites() {
    return [{
      source: "/api/:path*",
      destination: `${process.env.API_URL ?? "http://127.0.0.1:8000"}/:path*`,
    }];
  },
};

export default nextConfig;
