import { PDFLoader } from "@langchain/community/document_loaders/fs/pdf";
import { defaultTextSplitter } from "@langchain/core/messages";
import { ChatPromptTemplate } from "@langchain/core/prompts";
import { MistralAI, MistralAIEmbeddings } from "@langchain/mistralai";
import { create } from "domain";
import { config } from "dotenv";
import { RetrievalQAChain } from "langchain/chains";
import { createStuffDocumentsChain } from "langchain/chains/combine_documents";
import { createRetrievalChain } from "langchain/chains/retrieval";
import { CharacterTextSplitter } from "langchain/text_splitter";
import { MemoryVectorStore } from "langchain/vectorstores/memory";

config();
import path from "path";

class PdfQa {
  constructor({
    model,
    pdfDocument,
    chunkSize,
    chunkOverlap,
    kdocuments,
    searchType,
  }) {
    this.model = model || "codestral-latest";
    this.pdfDocument = pdfDocument;
    this.chunkSize = chunkSize || 1000;
    this.chunkOverlap = chunkOverlap || 200;
    this.kdocuments = kdocuments || 3;
    this.searchType = searchType || "similarity";
  }

  async init() {
    this.initChat();
    await this.documentLoader();
    await this.documentSplit();
    this.selectEmbedding = new MistralAIEmbeddings({ model: "mistral-embed" });
    await this.createVectordB();
    await this.createRetriver();
    this.chain = await this.createChain();
    return this;
  }

  async initChat() {
    console.log("initializing chat...");
    this.llm = new MistralAI({
      model: this.model,
      temperature: 0.1,
      maxRetries: 3,
    });

    const response = await this.llm.invoke("what's up?");
    console.log(response);
  }

  async documentLoader() {
    console.log("loading documents...");
    const loader = new PDFLoader(
      path.join(import.meta.dirname, this.pdfDocument)
    );
    this.docs = await loader.load();
    console.log(this.docs.length);
  }

  async documentSplit() {
    console.log("splitting documents...");
    const splitter = new CharacterTextSplitter({
      separator: " ",
      chunkSize: this.chunkSize,
      chunkOverlap: this.chunkOverlap,
    });

    this.texts = await splitter.splitDocuments(this.docs);
  }
  async createVectordB() {
    console.log("creating vectorDB...");
    this.db = await MemoryVectorStore.fromDocuments(
      this.texts,
      this.selectEmbedding
    );
  }

  async createRetriver() {
    console.log("creating retriever...");
    this.retriver = await this.db.asRetriever({
      k: this.kdocuments,
      searchType: this.searchType,
    });
  }

  async createChain() {
    console.log("creating chain...");
    const prompt = ChatPromptTemplate.fromTemplate(
      `Answer the user's question: {input} based on the following context {context}`
    );

    const combineDocsChain = await createStuffDocumentsChain({
      llm: this.llm,
      prompt,
    });

    const retrievalChain = await createRetrievalChain({
      combineDocsChain,
      retriever: this.retriver,
    });
    return retrievalChain;
  }
  querChain(query) {
    return this.chain;
  }
}

const pdfDocument = "./sample/tank.pdf";
const pdfQa = await new PdfQa({
  model: "codestral-latest",
  pdfDocument,
}).init();

const pdfChain = await pdfQa.querChain();
const query = "What is manual scrubbing?";
const response = await pdfChain.invoke({ input: query });
console.log(response.answer);

const chatHistory = [response.input, response.answer];
//console.log("chk22")
const query2 = "Why this manual scrubbing is tedious?";
const response2 = await pdfChain.invoke({ input: query2 });
console.log(response2.answer[1]);

//console.log(pdfQa.retriver.k)

// const related=await pdfQa.retriver.invoke("what is Manual scrubbing")
// console.log("Related documents:",related);

// console.log(pdfQa.db.memoryVectors.length);

// const similarityDoc= await pdfQa.db.similaritySearchWithScore("what is Manual scrubbing",3);
// console.log("Document pages and their score related to our query:");

// for( const [doc,score] of similarityDoc){
//     console.log(`Page Number: ${doc.metadata.pageNumber} Score: ${score}`);
//    // console.log(doc.pageContent);
// }
