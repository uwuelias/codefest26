import { useState, type ChangeEvent } from 'react';

const ALLOWED_FORMATS = ['wav', 'mp3', 'm4a'];

interface Props {
  onSubmit: (file: File) => void;
  loading?: boolean;
  error?: string | null;
}

export default function AudioSubmissionForm({ onSubmit, loading = false, error = null }: Props) {
  const [file, setFile] = useState<File | null>(null);
  const [fileError, setFileError] = useState<string | null>(null);

  const onPick = (e: ChangeEvent<HTMLInputElement>) => {
    const picked = e.target.files?.[0];
    e.target.value = '';
    if (!picked) return;

    const ext = picked.name.split('.').pop()?.toLowerCase().trim() ?? '';

    if (!ALLOWED_FORMATS.includes(ext)) {
      setFile(null);
      setFileError(`Unsupported format. Use ${ALLOWED_FORMATS.join(', ')}.`);
      return;
    }

    if (picked.size == 0) {
      setFile(null);
      setFileError('This file is empty.');
      return;
    }

    setFile(picked);
    setFileError(null);
  };

  const shownError = fileError ?? error;

  const handleSubmit = () => {
    if (file && !loading) onSubmit(file);
  };

  return (
    <div className="flex flex-col items-center w-full max-w-sm gap-4 py-4 rounded-xl border border-(--border) bg-(--background) text-(--color-text) ">
      <label htmlFor="audio-input" className="font-medium">
        Upload an audio clip
      </label>

      <input
        id="audio-input"
        type="file"
        onChange={onPick}
        disabled={loading}
        className="text-sm text-(--color-text-secondary) file:mr-3 file:rounded-md file:border file:border-(--border) file:bg-transparent file:px-3 file:py-1.5 file:text-sm file:text-(--color-text) hover:file:bg-neutral-100"
      />

      {file && <p className="text-sm text-(--color-text-secondary)">Selected: {file.name}</p>}

      <button
        type="button"
        onClick={handleSubmit}
        className="w-50 h-10 rounded-lg bg-(--color-text) px-4 py-5 text-sm font-medium text-white hover:opacity-90 disabled:cursor-not-allowed disabled:opacity-40 hover:cursor-pointer flex items-center justify-center"
      >
        {loading ? 'Analyzing…' : 'Check audio'}
      </button>

      {shownError && (
        <p role="alert" className="text-sm text-red-700">
          {shownError}
        </p>
      )}
    </div>
  );
}
