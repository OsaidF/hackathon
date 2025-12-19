import React, {
  createContext,
  useContext,
  useState,
  useEffect,
  ReactNode,
  useCallback,
} from 'react';
import ExecutionEnvironment from '@docusaurus/ExecutionEnvironment';
import toast from 'react-hot-toast';
import { z } from 'zod';

// API types
interface User {
  id: string;
  email: string;
  name?: string;
  image?: string;
  emailVerified: boolean;
  createdAt: string;
}

interface Session {
  id: string;
  token: string;
  expiresAt: string;
}

interface AuthResponse {
  user: User;
  session: Session;
}

interface AuthContextType {
  user: User | null;
  loading: boolean;
  isAuthenticated: boolean;
  login: (email: string, password: string) => Promise<void>;
  register: (email: string, password: string, name: string) => Promise<void>;
  logout: () => Promise<void>;
  checkAuth: () => Promise<void>;
  updateProfile: (data: { name?: string; image?: string | null }) => Promise<void>;
  changePassword: (currentPassword: string, newPassword: string) => Promise<void>;
  sendEmailVerification: () => Promise<void>;
}

const AuthContext = createContext<AuthContextType | undefined>(undefined);

interface AuthProviderProps {
  children: ReactNode;
  apiUrl?: string;
}

// Default API configuration
const DEFAULT_API_URL = 'https://osaid99-hackathon-serv.hf.space/api';

// Create API client factory
const createApiClient = (baseUrl: string) => ({
  async request<T>(
    endpoint: string,
    options: RequestInit = {}
  ): Promise<T> {
    // Safety check for SSR
    if (!ExecutionEnvironment.canUseDOM) {
      throw new Error('API calls are only available on the client side');
    }

    const url = `${baseUrl}${endpoint}`;
    const response = await fetch(url, {
      ...options,
      headers: {
        'Content-Type': 'application/json',
        ...options.headers,
      },
      credentials: 'include',
    });

    if (!response.ok) {
      const errorData = await response.json().catch(() => ({}));
      throw new Error(errorData.error?.message || 'API request failed');
    }

    return response.json() as Promise<T>;
  },

  async get<T>(endpoint: string): Promise<T> {
    return this.request<T>(endpoint, { method: 'GET' });
  },

  async post<T>(endpoint: string, data?: any): Promise<T> {
    return this.request<T>(endpoint, {
      method: 'POST',
      body: JSON.stringify(data),
    });
  },

  async put<T>(endpoint: string, data?: any): Promise<T> {
    return this.request<T>(endpoint, {
      method: 'PUT',
      body: JSON.stringify(data),
    });
  },

  async delete<T>(endpoint: string): Promise<T> {
    return this.request<T>(endpoint, { method: 'DELETE' });
  },
});

export const AuthProvider: React.FC<AuthProviderProps> = ({
  children,
  apiUrl = DEFAULT_API_URL
}) => {
  const [user, setUser] = useState<User | null>(null);
  const [loading, setLoading] = useState(true);
  const [mounted, setMounted] = useState(false);

  // Create API client with provided URL
  const apiClient = createApiClient(apiUrl);

  // Check authentication status
  const checkAuth = useCallback(async () => {
    // Skip on server-side
    if (!ExecutionEnvironment.canUseDOM) {
      return;
    }

    try {
      // Better-auth session endpoint: GET /api/auth/get-session
      const response = await apiClient.get<any>('/auth/get-session');

      // Log response to debug structure
      if (process.env.NODE_ENV === 'development') {
        console.log('Session response:', response);
      }

      // Handle different response structures
      // Better-auth typically returns: { user: User, session: Session } or null
      if (response && typeof response === 'object') {
        // Check if response has user directly
        if (response.user) {
          setUser(response.user);
        }
        // Check if response IS the user object
        else if (response.id && response.email) {
          setUser(response as User);
        }
        // No user found
        else {
          setUser(null);
        }
      } else {
        setUser(null);
      }
    } catch (error) {
      // Silently handle auth check failures (expected when not authenticated or backend down)
      // Only log in development
      if (process.env.NODE_ENV === 'development') {
        console.debug('Auth check failed (this is normal if not logged in):', error);
      }
      setUser(null);
    } finally {
      setLoading(false);
    }
  }, [apiUrl]);

  // Login function
  const login = useCallback(async (email: string, password: string): Promise<void> => {
    if (!ExecutionEnvironment.canUseDOM) {
      throw new Error('Login is only available on the client side');
    }

    try {
      setLoading(true);

      // Validate input
      const loginSchema = z.object({
        email: z.string().email('Invalid email address'),
        password: z.string().min(1, 'Password is required'),
      });

      const validatedData = loginSchema.parse({ email, password });

      // Better-auth sign-in endpoint: POST /api/auth/sign-in/email
      const response = await apiClient.post<{ user: User; session: Session }>(
        '/auth/sign-in/email', 
        validatedData
      );

      setUser(response.user);
      toast.success('Successfully logged in!');
    } catch (error) {
      console.error('Login failed:', error);
      toast.error(error instanceof Error ? error.message : 'Login failed');
      throw error;
    } finally {
      setLoading(false);
    }
  }, [apiUrl]);

  // Register function
  const register = useCallback(async (email: string, password: string, name: string): Promise<void> => {
    if (!ExecutionEnvironment.canUseDOM) {
      throw new Error('Registration is only available on the client side');
    }

    try {
      setLoading(true);

      // Validate input
      const registerSchema = z.object({
        email: z.string().email('Invalid email address'),
        password: z.string()
          .min(8, 'Password must be at least 8 characters long')
          .max(128, 'Password must be less than 128 characters'),
        name: z.string()
          .min(2, 'Name must be at least 2 characters long')
          .max(255, 'Name must be less than 255 characters'),
      });

      const validatedData = registerSchema.parse({ email, password, name });

      // Better-auth sign-up endpoint: POST /api/auth/sign-up/email
      const response = await apiClient.post<{ user: User; session: Session }>(
        '/auth/sign-up/email', 
        validatedData
      );

      setUser(response.user);
      toast.success('Successfully registered!');
    } catch (error) {
      console.error('Registration failed:', error);
      toast.error(error instanceof Error ? error.message : 'Registration failed');
      throw error;
    } finally {
      setLoading(false);
    }
  }, [apiUrl]);

  // Logout function
  const logout = useCallback(async (): Promise<void> => {
    if (!ExecutionEnvironment.canUseDOM) {
      throw new Error('Logout is only available on the client side');
    }

    try {
      setLoading(true);

      // Better-auth sign-out endpoint: POST /api/auth/sign-out
      await apiClient.post('/auth/sign-out', {});

      setUser(null);
      toast.success('Successfully logged out!');
    } catch (error) {
      console.error('Logout failed:', error);
      toast.error('Logout failed');
      throw error;
    } finally {
      setLoading(false);
    }
  }, [apiUrl]);

  // Update profile function
  const updateProfile = useCallback(async (data: { name?: string; image?: string | null }): Promise<void> => {
    if (!ExecutionEnvironment.canUseDOM) {
      throw new Error('Profile update is only available on the client side');
    }

    try {
      setLoading(true);

      // Better-auth update user endpoint: POST /api/auth/update-user
      const response = await apiClient.post<{ user: User }>(
        '/auth/update-user', 
        data
      );

      setUser(response.user);
      toast.success('Profile updated successfully!');
    } catch (error) {
      console.error('Profile update failed:', error);
      toast.error(error instanceof Error ? error.message : 'Profile update failed');
      throw error;
    } finally {
      setLoading(false);
    }
  }, [apiUrl]);

  // Change password function
  const changePassword = useCallback(async (currentPassword: string, newPassword: string): Promise<void> => {
    if (!ExecutionEnvironment.canUseDOM) {
      throw new Error('Password change is only available on the client side');
    }

    try {
      setLoading(true);

      // Better-auth change password endpoint: POST /api/auth/change-password
      await apiClient.post('/auth/change-password', {
        currentPassword,
        newPassword,
      });

      toast.success('Password changed successfully!');
    } catch (error) {
      console.error('Password change failed:', error);
      toast.error(error instanceof Error ? error.message : 'Password change failed');
      throw error;
    } finally {
      setLoading(false);
    }
  }, [apiUrl]);

  // Send email verification function
  const sendEmailVerification = useCallback(async (): Promise<void> => {
    if (!ExecutionEnvironment.canUseDOM) {
      throw new Error('Email verification is only available on the client side');
    }

    try {
      setLoading(true);

      // Better-auth send verification email endpoint: POST /api/auth/send-verification-email
      await apiClient.post('/auth/send-verification-email', {
        email: user?.email,
      });

      toast.success('Verification email sent!');
    } catch (error) {
      console.error('Send email verification failed:', error);
      toast.error(error instanceof Error ? error.message : 'Failed to send verification email');
      throw error;
    } finally {
      setLoading(false);
    }
  }, [apiUrl, user?.email]);

  // Track mounting for hydration consistency
  useEffect(() => {
    setMounted(true);
  }, []);

  // Initialize auth state only on client-side after mount
  useEffect(() => {
    if (mounted && ExecutionEnvironment.canUseDOM) {
      checkAuth();
    }
  }, [mounted, checkAuth]);

  // Return loading=true until mounted to ensure consistent hydration
  const value: AuthContextType = {
    user,
    loading: !mounted || loading,
    isAuthenticated: mounted && !!user,
    login,
    register,
    logout,
    checkAuth,
    updateProfile,
    changePassword,
    sendEmailVerification,
  };

  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>;
};

export const useAuth = (): AuthContextType => {
  const context = useContext(AuthContext);
  if (context === undefined) {
    throw new Error('useAuth must be used within an AuthProvider');
  }
  return context;
};

export default AuthProvider;