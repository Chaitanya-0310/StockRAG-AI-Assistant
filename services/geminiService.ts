import { api } from './api';
import { ChatMessage } from '../types';

export const generateRAGResponse = async (
  query: string,
  history: ChatMessage[]
): Promise<{ text: string; contexts: any[] }> => {
  try {
    // The backend handles RAG internally now
    const response = await api.post('/chat', {
      query: query,
      history: history.map(h => ({ role: h.role, parts: [{ text: h.content }] }))
    });

    return {
      text: response.text,
      contexts: [] // Backend might not return contexts explicitly in the simple response, or we can add it
    };

  } catch (error) {
    console.error("RAG API Error:", error);
    return {
      text: "I encountered an error connecting to the AI analyst. Please ensure the backend is running.",
      contexts: []
    };
  }
};