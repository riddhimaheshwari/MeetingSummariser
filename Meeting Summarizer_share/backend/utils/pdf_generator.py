from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter

def create_mom_pdf(filename, full_text: str):
    c = canvas.Canvas(filename, pagesize=letter)
    width, height = letter

    c.setFont("Helvetica-Bold", 14)
    c.drawString(50, height - 50, "Meeting Summary and Notes")

    c.setFont("Helvetica", 10)
    y = height - 80
    for line in full_text.splitlines():
        if y < 50:
            c.showPage()
            y = height - 50
            c.setFont("Helvetica", 10)
        c.drawString(50, y, line.strip())
        y -= 15

    c.save()
