import React from 'react';
import Layout from '@theme-original/Layout';
import AskDocsButton from '@site/src/components/RAGChat';

export default function LayoutWrapper(props) {
  return (
    <>
      <Layout {...props} />
      <AskDocsButton />
    </>
  );
}