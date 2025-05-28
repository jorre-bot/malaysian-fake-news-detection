import React, { useState } from 'react';
import axios from 'axios';

function App() {
  const [text, setText] = useState('');
  const [result, setResult] = useState<any>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setLoading(true);
    setError(null);
    
    try {
      const response = await axios.post('http://localhost:5000/api/detect', { text });
      setResult(response.data);
    } catch (err: any) {
      setError(err.response?.data?.error || 'Something went wrong');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-gray-100 py-6 flex flex-col justify-center sm:py-12">
      <div className="relative py-3 sm:max-w-xl sm:mx-auto">
        <div className="absolute inset-0 bg-gradient-to-r from-cyan-400 to-light-blue-500 shadow-lg transform -skew-y-6 sm:skew-y-0 sm:-rotate-6 sm:rounded-3xl"></div>
        <div className="relative px-4 py-10 bg-white shadow-lg sm:rounded-3xl sm:p-20">
          <div className="max-w-md mx-auto">
            <div className="divide-y divide-gray-200">
              <div className="py-8 text-base leading-6 space-y-4 text-gray-700 sm:text-lg sm:leading-7">
                <h1 className="text-3xl font-bold text-center mb-8">Malaysian Fake News Detector</h1>
                <form onSubmit={handleSubmit} className="space-y-4">
                  <div>
                    <textarea
                      className="w-full px-3 py-2 text-gray-700 border rounded-lg focus:outline-none"
                      rows={4}
                      value={text}
                      onChange={(e) => setText(e.target.value)}
                      placeholder="Paste your news article here..."
                    />
                  </div>
                  <button
                    type="submit"
                    disabled={loading}
                    className="w-full px-4 py-2 text-white bg-blue-500 rounded-lg hover:bg-blue-600 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-opacity-50 disabled:opacity-50"
                  >
                    {loading ? 'Analyzing...' : 'Detect Fake News'}
                  </button>
                </form>

                {error && (
                  <div className="mt-4 p-4 text-red-700 bg-red-100 rounded-lg">
                    {error}
                  </div>
                )}

                {result && (
                  <div className="mt-8 p-6 bg-gray-50 rounded-lg">
                    <h2 className="text-xl font-semibold mb-4">Analysis Result</h2>
                    <div className="space-y-2">
                      <p className="flex justify-between">
                        <span>Prediction:</span>
                        <span className={`font-bold ${result.prediction === 'REAL' ? 'text-green-600' : 'text-red-600'}`}>
                          {result.prediction}
                        </span>
                      </p>
                      <p className="flex justify-between">
                        <span>Confidence:</span>
                        <span className="font-bold">{(result.confidence * 100).toFixed(2)}%</span>
                      </p>
                      <p className="flex justify-between">
                        <span>Sentiment:</span>
                        <span className="font-bold">{result.analysis.sentiment}</span>
                      </p>
                    </div>
                  </div>
                )}
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

export default App; 