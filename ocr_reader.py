import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext
from PIL import Image, ImageTk, ImageDraw, ImageFont
import torch
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
import os

# YOUR TRAINED MODEL
MODEL_DIR = r"C:\Users\Bless\Desktop\trocr-exam-project\models\trocr-exam\checkpoint-120"

class OCRReader:
    def __init__(self, root):
        self.root = root
        self.root.title("✍️ Handwritten Text Reader")
        self.root.geometry("1000x700")
        self.root.configure(bg="#f0f0f0")
        
        self.processor = None
        self.model = None
        self.device = None
        self.original_image = None
        self.image_path = None  # ✅ FIXED: Store path separately
        
        self.setup_ui()
        self.load_model()
    
    def setup_ui(self):
        # Title
        title = tk.Label(self.root, text="✍️ Handwritten Exam Reader", 
                        font=("Arial", 20, "bold"), bg="#2c3e50", fg="white")
        title.pack(pady=10, fill="x")
        
        # Main frame
        main_frame = tk.Frame(self.root, bg="white")
        main_frame.pack(fill="both", expand=True, padx=20, pady=10)
        
        # Left side - Image
        left_frame = tk.Frame(main_frame, bg="white")
        left_frame.pack(side="left", fill="both", expand=True)
        
        tk.Label(left_frame, text="📁 Upload Handwritten Image", 
                font=("Arial", 14, "bold"), bg="white").pack(pady=10)
        
        self.upload_btn = tk.Button(left_frame, text="📁 Choose Image", 
                                   command=self.upload_image, font=("Arial", 14),
                                   bg="#3498db", fg="white", height=2, width=20)
        self.upload_btn.pack(pady=20)
        
        self.image_label = tk.Label(left_frame, bg="#e8e8e8", width=60, height=25,
                                   text="No image loaded", font=("Arial", 12))
        self.image_label.pack(pady=10)
        
        self.extract_btn = tk.Button(left_frame, text="🔮 EXTRACT TEXT", 
                                   command=self.extract_text, font=("Arial", 16, "bold"),
                                   bg="#e74c3c", fg="white", height=2, width=20, 
                                   state="disabled")
        self.extract_btn.pack(pady=20)
        
        # Right side - Text result
        right_frame = tk.Frame(main_frame, bg="#ecf0f1", width=400)
        right_frame.pack(side="right", fill="y", padx=(10,0))
        right_frame.pack_propagate(False)
        
        tk.Label(right_frame, text="📄 Extracted Digital Text", 
                font=("Arial", 16, "bold"), bg="#ecf0f1").pack(pady=(20,10))
        
        self.text_area = scrolledtext.ScrolledText(right_frame, height=20, width=40, 
                                                 font=("Consolas", 14), wrap=tk.WORD,
                                                 state="disabled", bg="#ffffff")
        self.text_area.pack(fill="both", expand=True, pady=10, padx=15)
        
        self.copy_btn = tk.Button(right_frame, text="📋 Copy to Clipboard", 
                                 command=self.copy_text, font=("Arial", 12, "bold"),
                                 bg="#27ae60", fg="white", height=2, state="disabled")
        self.copy_btn.pack(pady=15)
    
    def load_model(self):
        """Load trained model"""
        try:
            print("🔄 Loading your TrOCR model...")
            self.processor = TrOCRProcessor.from_pretrained(MODEL_DIR, local_files_only=True)
            self.model = VisionEncoderDecoderModel.from_pretrained(MODEL_DIR, local_files_only=True)
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            self.model.to(self.device)
            self.model.eval()
            print(f"✅ Model loaded on {self.device}")
        except Exception as e:
            messagebox.showerror("❌ Model Error", f"Cannot load model:\n{str(e)}")
            self.root.quit()
    
    def upload_image(self):
        """Upload handwritten image"""
        file_path = filedialog.askopenfilename(
            title="Select Handwritten Image",
            filetypes=[("Image Files", "*.jpg *.jpeg *.png *.bmp *.tiff")]
        )
        
        if file_path:
            try:
                # ✅ FIXED: Store path AND image separately
                self.image_path = file_path
                self.original_image = Image.open(file_path).convert("RGB")
                
                # Resize for display
                display_img = self.original_image.copy()
                display_img.thumbnail((450, 450))
                
                # Show image
                photo = ImageTk.PhotoImage(display_img)
                self.image_label.configure(image=photo, text="")
                self.image_label.image = photo  # Keep reference
                
                # Enable extract button
                self.extract_btn.config(state="normal")
                
                # Clear previous text
                self.text_area.config(state="normal")
                self.text_area.delete(1.0, tk.END)
                self.text_area.insert(tk.END, "✓ Image ready!\nClick EXTRACT TEXT")
                self.text_area.config(state="disabled")
                self.copy_btn.config(state="disabled")
                
                print(f"✅ Image loaded: {os.path.basename(file_path)}")
                
            except Exception as e:
                messagebox.showerror("❌ Image Error", f"Cannot load image:\n{str(e)}")
    
    def extract_text(self):
        """Extract handwritten text"""
        if not self.original_image or not self.model:
            messagebox.showwarning("⚠️ Warning", "Please upload image first!")
            return
        
        try:
            self.extract_btn.config(state="disabled", text="🔄 Extracting...")
            self.root.update()
            
            print("🔮 Predicting...")
            
            # Process original image (full quality)
            inputs = self.processor(images=self.original_image, return_tensors="pt").to(self.device)
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_length=128,
                    num_beams=4,
                    do_sample=False,
                    early_stopping=True,
                    pad_token_id=self.processor.tokenizer.pad_token_id
                )
            
            # Decode text
            extracted_text = self.processor.decode(outputs[0], skip_special_tokens=True)
            
            # Display result BIG AND CLEAR
            self.text_area.config(state="normal")
            self.text_area.delete(1.0, tk.END)
            
            self.text_area.insert(tk.END, f"🎯 EXTRACTED TEXT:\n")
            self.text_area.insert(tk.END, "="*50 + "\n")
            self.text_area.insert(tk.END, f'"{extracted_text}"\n')
            self.text_area.insert(tk.END, "="*50 + "\n\n")
            self.text_area.insert(tk.END, f"📏 Length: {len(extracted_text)} chars")
            
            self.text_area.config(state="disabled")
            self.copy_btn.config(state="normal")
            
            # Save result image with text overlay
            result_img = self.original_image.copy()
            draw = ImageDraw.Draw(result_img)
            try:
                font = ImageFont.truetype("arial.ttf", 50)
            except:
                font = ImageFont.load_default()
            
            # Draw yellow background + black text
            bbox = draw.textbbox((20, 20), extracted_text, font=font)
            draw.rectangle([bbox[0]-10, bbox[1]-10, bbox[2]+10, bbox[3]+10], 
                          fill=(255, 255, 0, 200))
            draw.text((20, 20), extracted_text, fill=(0, 0, 0), font=font)
            result_img.save("extracted_result.jpg")
            
            print(f"✅ SUCCESS: '{extracted_text}'")
            print(f"💾 Saved: extracted_result.jpg")
            
        except Exception as e:
            messagebox.showerror("❌ Error", f"Extraction failed:\n{str(e)}")
            print(f"❌ Error: {e}")
        finally:
            self.extract_btn.config(state="normal", text="🔮 EXTRACT TEXT")
    
    def copy_text(self):
        """Copy extracted text to clipboard"""
        text = self.text_area.get(1.0, tk.END).strip()
        self.root.clipboard_clear()
        self.root.clipboard_append(text)
        messagebox.showinfo("✅ Copied!", "Text copied to clipboard!")

if __name__ == "__main__":
    root = tk.Tk()
    app = OCRReader(root)
    root.mainloop()
