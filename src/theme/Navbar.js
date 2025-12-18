import Navbar from '@theme-original/Navbar';
import { useAuth } from '../components/Auth/AuthProvider';
import styles from './navbar.module.css';

export default function NavbarWrapper(props) {
  try {
    const { user, loading, logout } = useAuth();

    if (loading) {
      return <Navbar {...props} />;
    }

    return (
      <>
        <Navbar {...props} />
      </>
    );
  } catch (error) {
    console.error('Auth error in Navbar:', error);
    return <Navbar {...props} />;
  }
}