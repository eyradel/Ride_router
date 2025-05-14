import axios from "axios";

const API_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

const api = axios.create({
  baseURL: API_URL,
  headers: {
    "Content-Type": "application/json",
  },
});

// Add a request interceptor to include the token in requests
api.interceptors.request.use((config) => {
  const token = localStorage.getItem("token");
  if (token) {
    config.headers.Authorization = `Bearer ${token}`;
  }
  return config;
});

// Add a response interceptor to handle token expiration
api.interceptors.response.use(
  (response) => response,
  async (error) => {
    if (error.response?.status === 401) {
      localStorage.removeItem("token");
      window.location.href = "/login";
    }
    return Promise.reject(error);
  }
);

export const auth = {
  login: async (email: string, password: string) => {
    try {
      // Create form data for OAuth2 password flow
      const formData = new URLSearchParams();
      formData.append("grant_type", "password");
      formData.append("username", email);
      formData.append("password", password);

      console.log("Attempting login with:", { email }); // Debug log

      const response = await axios.post(`${API_URL}/auth/token`, formData.toString(), {
        headers: {
          "Content-Type": "application/x-www-form-urlencoded",
          "Accept": "application/json",
        },
      });

      console.log("Login response:", response.data); // Debug log

      if (response.data && response.data.access_token) {
        localStorage.setItem("token", response.data.access_token);
        return response.data;
      } else {
        throw new Error("No access token received");
      }
    } catch (error: any) {
      console.error("Login error:", error.response?.data || error); // Debug log
      
      if (error.response?.data?.detail) {
        throw new Error(error.response.data.detail);
      }
      throw new Error("Login failed. Please check your credentials.");
    }
  },

  register: async (data: any) => {
    try {
      const response = await api.post("/auth/register", data);
      return response.data;
    } catch (error: any) {
      if (error.response?.data?.detail) {
        throw new Error(error.response.data.detail);
      }
      throw new Error("Registration failed. Please try again.");
    }
  },

  verify: async (token: string) => {
    const response = await api.get(`/auth/verify?token=${token}`);
    return response.data;
  },

  verifyToken: async () => {
    try {
      const token = localStorage.getItem("token");
      if (!token) return false;

      const response = await api.get("/verify-token", {
        headers: {
          Authorization: `Bearer ${token}`,
        },
      });
      return response.status === 200;
    } catch (error) {
      localStorage.removeItem("token");
      return false;
    }
  },

  logout: () => {
    localStorage.removeItem("token");
    window.location.href = "/login";
  },

  getProfile: async () => {
    const response = await api.get("/users/me");
    return response.data;
  },
};

export const routes = {
  optimize: async (data: any) => {
    const response = await api.post("/optimize", data);
    return response.data;
  },

  getHistory: async () => {
    const response = await api.get("/history");
    return response.data;
  },

  getRoute: async (id: string) => {
    const response = await api.get(`/routes/${id}`);
    return response.data;
  },
};

export default api; 