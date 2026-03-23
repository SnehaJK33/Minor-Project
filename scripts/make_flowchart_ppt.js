const pptxgen = require('pptxgenjs');

const pptx = new pptxgen();
pptx.layout = 'LAYOUT_WIDE';

const slide = pptx.addSlide();

// Title
slide.addText('Project Workflow', {
  x: 0.5,
  y: 0.3,
  w: 12.33,
  h: 0.6,
  fontSize: 32,
  bold: true,
  color: '1A1A1A',
  align: 'center',
});

const boxes = [
  { label: 'Data\nCollection', color: '1B6F7A' },
  { label: 'Data\nPreprocessing', color: '3F8F3F' },
  { label: 'Feature\nEngineering', color: '8B9F3A' },
  { label: 'Clustering', color: 'D9A441' },
  { label: 'Prediction', color: '2B6CB0' },
  { label: 'Results &\nDashboards', color: '1F6F63' },
];

const boxW = 2.0;
const boxH = 1.5;
const gap = 0.25;
const startX = 0.05;
const y = 3.1;

boxes.forEach((b, i) => {
  const x = startX + i * (boxW + gap);
  slide.addShape(pptx.ShapeType.roundRect, {
    x,
    y,
    w: boxW,
    h: boxH,
    fill: { color: b.color },
    line: { color: b.color },
    radius: 0.08,
  });
  slide.addText(b.label, {
    x,
    y,
    w: boxW,
    h: boxH,
    fontSize: 18,
    color: 'FFFFFF',
    bold: true,
    align: 'center',
    valign: 'middle',
  });

  // Arrow to next box
  if (i < boxes.length - 1) {
    const arrowX = x + boxW;
    const arrowY = y + boxH / 2;
    slide.addShape(pptx.ShapeType.line, {
      x: arrowX,
      y: arrowY,
      w: gap,
      h: 0,
      line: { color: '555555', width: 2, endArrowType: 'triangle' },
    });
  }
});

pptx.writeFile({ fileName: 'project_workflow.pptx' });
