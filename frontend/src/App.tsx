import AudioSubmissionForm from './components/AudioSubmissionForm';
import { useState } from 'react';

interface IdentifyResult {
  verdict: 'real' | 'fake' | 'inconclusive';
  confidence: number;
  p_fake: number;
}

const configuredApiUrl = import.meta.env.VITE_API_URL;
const API_URL = /^https?:\/\//.test(configuredApiUrl)
  ? configuredApiUrl
  : `http://${configuredApiUrl}`;

function App() {
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<IdentifyResult | null>(null);

  const handleSubmit = async (file: File) => {
    setLoading(true);
    setError(null);
    setResult(null);

    try {
      const body = new FormData();
      body.append('file', file);

      const res = await fetch(`${API_URL}/api/predict`, { method: 'POST', body });

      if (!res.ok) {
        const data = await res.json().catch(() => null);
        throw new Error(data?.detail ?? `Request failed (${res.status})`);
      }

      setResult(await res.json());
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Something went wrong.');
    } finally {
      setLoading(false);
    }
  };

  return (
    <>
      <div className="flex flex-col items-center justify-center h-screen gap-3">
        <h1 className="text-3xl font-bold text-(--color-text)">Heavenly King Sound</h1>
        <p className="text-(--color-text-muted)">
          Upload a short recording or record. We'll tell you the verdict and the confidence score.
        </p>
        <AudioSubmissionForm onSubmit={handleSubmit} loading={loading} error={error} />
        {result && (
          <div role="status" className="text-(--color-text)">
            <p className="text-xl font-semibold">
              {result.verdict === 'fake' && 'Likely AI-generated'}
              {result.verdict === 'real' && 'Likely human'}
              {result.verdict === 'inconclusive' && 'Not sure'}
            </p>
            <p className="text-(--color-text-muted)">
              {Math.round(result.confidence * 100)}% confident
            </p>
          </div>
        )}
      </div>
    </>
  );
}

export default App;
