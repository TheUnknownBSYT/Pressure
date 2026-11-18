"""Render the shared methodology blocks to editable Word. Run from the build directory."""
import json
from pathlib import Path

from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_CELL_VERTICAL_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor

OUT=Path('outputs/final_submission')


def ranked_table(doc, headers, rows, widths):
    table=doc.add_table(rows=0,cols=len(headers))
    table.autofit=False
    table.alignment=WD_TABLE_ALIGNMENT.CENTER
    for col, width in zip(table.columns,widths):
        col.width=Inches(width)
    for j, row in enumerate([headers]+rows):
        cells=table.add_row().cells
        for k,(cell,label,w) in enumerate(zip(cells,row,widths)):
            cell.width=Inches(w)
            cell.vertical_alignment=WD_CELL_VERTICAL_ALIGNMENT.CENTER
            fill=OxmlElement('w:shd')
            fill.set(qn('w:fill'),'21483F' if not j else ('EBF1ED' if j%2 else 'F9FAF8'))
            cell._tc.get_or_add_tcPr().append(fill)
            p=cell.paragraphs[0]
            p.paragraph_format.space_before=Pt(4)
            p.paragraph_format.space_after=Pt(4)
            p.paragraph_format.line_spacing=1.02
            if k>1: p.alignment=WD_ALIGN_PARAGRAPH.RIGHT
            r=p.add_run(str(label));r.font.size=Pt(9.5)
            r.bold=not j
            if not j:r.font.color.rgb=RGBColor.from_string('FFFFFF')


def main():
    pages=json.loads((OUT/'evidence/report_content.json').read_text())
    doc=Document()
    sect=doc.sections[0]
    sect.page_width=Inches(8.5);sect.page_height=Inches(11)
    sect.left_margin=sect.right_margin=Inches(.82)
    sect.top_margin=Inches(.82);sect.bottom_margin=Inches(.7)
    sect.header_distance=Inches(.36);sect.footer_distance=Inches(.32)
    for name,size in [('Normal',11.2),('Title',30),('Heading 1',24),('Heading 2',14),('Caption',9.3)]:
        s=doc.styles[name];s.font.name='Calibri';s.font.size=Pt(size)
        s.font.color.rgb=RGBColor.from_string('121917' if name=='Normal' else '000000')
        fmt=s.paragraph_format
        fmt.space_before=Pt(9 if name=='Heading 2' else 0)
        fmt.space_after=Pt(7 if name!='Caption' else 5);fmt.line_spacing=1.12
        if name=='Caption': s.font.italic=True
    props=doc.styles['Title']._element.get_or_add_pPr()
    for el in props.findall(qn('w:pBdr')):props.remove(el)
    head=sect.header.paragraphs[0]
    head.alignment=WD_ALIGN_PARAGRAPH.RIGHT
    r=head.add_run('SUPERR SELECTOR   /   SEASON 2   /   ROUND 1')
    r.font.size=Pt(8);r.font.color.rgb=RGBColor.from_string('51635A');r.bold=True
    footer=sect.footer.paragraphs[0]
    footer.paragraph_format.tab_stops.add_tab_stop(Inches(6.45))
    r=footer.add_run('T20 performance under pressure                                      18 November 2026\t')
    r.font.size=Pt(9)
    field=OxmlElement('w:fldSimple');field.set(qn('w:instr'),'PAGE');footer._p.append(field)
    for i,(title,blocks) in enumerate(pages):
        if i:doc.add_page_break()
        doc.add_paragraph(title,'Title' if not i else 'Heading 1')
        for typ,*args in blocks:
            if typ=='table':ranked_table(doc,*args)
            elif typ=='image':
                doc.add_picture(str(OUT/args[0]),width=Inches(6.65))
                doc.add_paragraph(args[1],'Caption')
            else:doc.add_paragraph(args[0],{'p':'Normal','h':'Heading 2','cap':'Caption'}[typ])
    doc.core_properties.title='T20 performance under pressure'
    doc.core_properties.subject='SupeRR Selector Season 2 Round 1 methodology and ordered selections'
    doc.save(OUT/'SupeRR_Selector_Final_Methodology.docx')


if __name__=='__main__':main()
