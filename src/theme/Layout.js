import Layout from '@theme-original/Layout';
import AskDocsButton from '@site/src/components/RAGChat';
import { useLocation } from '@docusaurus/router';


export default function LayoutWrapper(props) {
  const location = useLocation();
    
  const isDocsPage = location.pathname.startsWith('/docs') || 
                    location.pathname.startsWith('/api') ||
                    location.pathname.startsWith('/guides');
  return (
    <>
    <Layout {...props} />
    {isDocsPage && <AskDocsButton />}
         
    </>
  );
}