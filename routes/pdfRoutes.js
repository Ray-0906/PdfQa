import express from 'express';
import multer from 'multer';
import { pdfController } from '../controllers/pdfControllers.js';


const router = express.Router();
const upload = multer({ dest: 'uploads/' });

router.post('/upload', upload.single('pdf'), async (req, res) => {
  try {
    await pdfController.processPDF(req.file.path);
    res.status(200).json({ message: 'PDF processed successfully' });
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});

router.post('/query', async (req, res) => {
  try {
    const result = await pdfController.queryPDF(req.body.question);
    res.status(200).json(result);
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});

router.get('/history', async (req, res) => {
  try {
    res.status(200).json(pdfController.chatHistory);
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});

router.get('/pdfs', async (req, res) => {
    try {
      // In a real app, you'd get this from MongoDB
      const pdfs = []; // Replace with actual DB query
      res.status(200).json(pdfs);
    } catch (error) {
      res.status(500).json({ error: error.message });
    }
  });
export default router;