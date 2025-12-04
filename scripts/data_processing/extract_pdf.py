import PyPDF2
import sys

def extract_pdf_text(pdf_path, output_path, encoding='utf-8'):
    """PDFからテキストを抽出する"""
    try:
        with open(pdf_path, 'rb') as pdf_file:
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            total_pages = len(pdf_reader.pages)
            print(f"総ページ数: {total_pages}")
            
            # 最初の3ページのみを確認用に抽出
            sample_text = ""
            for i in range(min(3, total_pages)):
                page = pdf_reader.pages[i]
                text = page.extract_text()
                sample_text += f"\n\n=== Page {i+1} ===\n{text}"
            
            # サンプルを表示
            print(sample_text[:1000])  # 最初の1000文字
            
            # 全ページを抽出してファイルに保存
            full_text = ""
            for i in range(total_pages):
                page = pdf_reader.pages[i]
                full_text += page.extract_text() + "\n\n"
            
            # UTF-8で保存
            with open(output_path, 'w', encoding=encoding) as f:
                f.write(full_text)
            
            print(f"\n\n抽出完了: {output_path} ({encoding})")
            return True
            
    except Exception as e:
        print(f"エラー: {e}")
        return False

if __name__ == "__main__":
    pdf_path = r'C:\Users\socce\software-lab\traffic-accident\data\raw\codebook_2024.pdf'
    output_path = r'C:\Users\socce\software-lab\traffic-accident\data\codebook\codebook_2024_extracted.txt'
    extract_pdf_text(pdf_path, output_path)

