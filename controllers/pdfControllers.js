import { PDFLoader } from "@langchain/community/document_loaders/fs/pdf";
import { MistralAI, MistralAIEmbeddings } from "@langchain/mistralai";
import { CharacterTextSplitter } from "langchain/text_splitter";
import { MemoryVectorStore } from "langchain/vectorstores/memory";
import { ChatPromptTemplate } from "@langchain/core/prompts";
import { createStuffDocumentsChain } from "langchain/chains/combine_documents";
import { createRetrievalChain } from "langchain/chains/retrieval";

class PDFController {
  constructor() {
    this.retrievalChain = null;
    this.chatHistory = [];
  }

  async processPDF(filePath) {
    try {
      // 1. Load PDF
      const loader = new PDFLoader(filePath);
      const docs = await loader.load();
      
      // 2. Split documents
      const splitter = new CharacterTextSplitter({
        chunkSize: 1000,
        chunkOverlap: 200,
      });
      const texts = await splitter.splitDocuments(docs);
      
      // 3. Create vector store
      const embeddings = new MistralAIEmbeddings({ model: "mistral-embed" });
      const vectorStore = await MemoryVectorStore.fromDocuments(texts, embeddings);
      
      // 4. Initialize LLM
      const llm = new MistralAI({
        model: "codestral-latest",
        temperature: 0.1,
      });
      
      // 5. Create the chain (using your preferred non-deprecated approach)
      const prompt = ChatPromptTemplate.fromTemplate(`
        Answer the question based only on the following context:
        {context}
        
        Question: {input}
        
        Chat History: {chat_history}
      `);
      
      const combineDocsChain = await createStuffDocumentsChain({
        llm,
        prompt,
      });
      
      const retriever = vectorStore.asRetriever({
        k: 3,
        searchType: "similarity",
      });
      
      this.retrievalChain = await createRetrievalChain({
        combineDocsChain,
        retriever,
      });
      
      return true;
    } catch (error) {
      throw new Error(`PDF processing failed: ${error.message}`);
    }
  }

  async queryPDF(question) {
    if (!this.retrievalChain) throw new Error("PDF not processed yet");
    
    const response = await this.retrievalChain.invoke({
      input: question,
      chat_history: this.chatHistory,
    });
    
    // Update chat history
    this.chatHistory.push({
      question,
      answer: response.answer,
    });
    
    return {
      answer: response.answer,
      sources: response.context.map(doc => ({
        page: doc.metadata.pageNumber,
        content: doc.pageContent,
      })),
    };
  }
}

export const pdfController = new PDFController();