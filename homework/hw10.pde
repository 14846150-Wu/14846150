// 水平彈簧伸縮 - 直線+波浪+直線結構
// 牆壁 → 直線 → 彈簧波浪 → 直線 → 方塊

// === 運動參數 ===
float amplitude = 100;          // 振幅
float frequency = 0.04;         // 頻率
float damping = 0.998;          // 阻尼係數
float phase = PI;               // 初始相位
float time = 0;

// === 結構參數 ===
float wallX = 100;              // 牆壁X位置
float lineLength = 80;          // 連接直線長度
float springRestLength = 250;   // 彈簧靜止長度
float springCoils = 18;         // 彈簧圈數
float springAmplitude = 40;     // 彈簧波浪高度
float springY;                  // 彈簧Y位置

// === 方塊參數 ===
float blockSize = 80;
float blockX;

// === 波形圖 ===
ArrayList<Float> waveHistory = new ArrayList<Float>();
int maxHistoryLength = 400;

// === 色彩 ===
color bgColor = color(180);
color wallColor = color(255);
color lineColor = color(255);
color springColor = color(255);
color blockColor = color(0, 0, 255);
color waveColor = color(255);

// === 控制 ===
boolean isPaused = false;
boolean showInfo = true;
boolean showWave = true;

void setup() {
  size(900, 600);
  springY = height / 2;
  frameRate(30);
  
  for (int i = 0; i < maxHistoryLength; i++) {
    waveHistory.add(0.0);
  }
  
  println("=== 水平彈簧伸縮模擬器 ===");
  println("結構：牆壁 → 直線 → 彈簧波浪 → 直線 → 方塊");
  println("按 H 鍵查看幫助");
}

void draw() {
  background(bgColor);
  
  // 計算位移
  float omega = TWO_PI * frequency;
  float currentAmp = amplitude * pow(damping, time / 10.0);
  float displacement = currentAmp * sin(omega * time + phase);
  
  // 計算各部分位置
  float line1Start = wallX;
  float line1End = wallX + lineLength;
  
  float springStart = line1End;
  float springEnd = springStart + springRestLength + displacement;
  
  float line2Start = springEnd;
  float line2End = line2Start + lineLength;
  
  blockX = line2End + blockSize/2;
  
  // 更新歷史
  if (!isPaused) {
    waveHistory.remove(0);
    waveHistory.add(displacement);
    time += 1.0;
  }
  
  // === 繪製 ===
  drawReference();
  drawWall();
  drawLine1();
  drawSpring(springStart, springEnd);
  drawLine2(line2Start, line2End);
  drawBlock();
  
  if (showWave) {
    drawWaveform();
  }
  
  if (showInfo) {
    drawInfoPanel();
  }
  
  // 重置
  if (currentAmp < 3 && !isPaused) {
    resetAnimation();
  }
}

// 繪製參考線
void drawReference() {
  stroke(200, 200, 220);
  strokeWeight(1);
  float refX = wallX + lineLength + springRestLength + lineLength;
  
  // 平衡位置虛線
  for (int y = 0; y < height; y += 10) {
    point(refX, y);
  }
  
  // 標示
  fill(200);
  textAlign(CENTER);
  textSize(11);
  pushMatrix();
  translate(refX - 15, springY);
  rotate(-HALF_PI);
  text("平衡位置", 0, 0);
  popMatrix();
}

// 繪製牆壁（L形）
void drawWall() {
  stroke(wallColor);
  strokeWeight(5);
  
  // 垂直線
  line(wallX, 50, wallX, height - 50);
  
  // 水平支架（上）
  line(wallX, springY - 100, wallX + 30, springY - 100);
  
  // 水平支架（下）
  line(wallX, springY + 100, wallX + 30, springY + 100);
}

// 繪製第一段直線（牆壁到彈簧）
void drawLine1() {
  stroke(lineColor);
  strokeWeight(3);
  line(wallX, springY, wallX + lineLength, springY);
  
  // 端點
  fill(lineColor);
  noStroke();
  ellipse(wallX, springY, 8, 8);
  ellipse(wallX + lineLength, springY, 8, 8);
}

// 繪製彈簧波浪
void drawSpring(float startX, float endX) {
  float springLength = endX - startX;
  float coilWidth = springLength / springCoils;
  
  // 陰影
  stroke(0, 0, 0, 40);
  strokeWeight(4);
  noFill();
  
  beginShape();
  vertex(startX + 2, springY + 2);
  for (int i = 1; i < springCoils; i++) {
    float x = startX + i * coilWidth + 2;
    float y = springY + 2 + (i % 2 == 1 ? -springAmplitude : springAmplitude);
    vertex(x, y);
  }
  vertex(endX + 2, springY + 2);
  endShape();
  
  // 彈簧主體
  stroke(springColor);
  strokeWeight(4);
  
  beginShape();
  vertex(startX, springY);
  for (int i = 1; i < springCoils; i++) {
    float x = startX + i * coilWidth;
    float y = springY + (i % 2 == 1 ? -springAmplitude : springAmplitude);
    vertex(x, y);
  }
  vertex(endX, springY);
  endShape();
  
  // 彈簧端點
  fill(springColor);
  noStroke();
  ellipse(startX, springY, 10, 10);
  ellipse(endX, springY, 10, 10);
}

// 繪製第二段直線（彈簧到方塊）
void drawLine2(float startX, float endX) {
  stroke(lineColor);
  strokeWeight(3);
  line(startX, springY, endX, springY);
  
  // 端點
  fill(lineColor);
  noStroke();
  ellipse(startX, springY, 8, 8);
  ellipse(endX, springY, 8, 8);
}

// 繪製藍色方塊
void drawBlock() {
  // 陰影
  fill(0, 0, 0, 50);
  noStroke();
  rect(blockX - blockSize/2 + 4, springY - blockSize/2 + 4, 
       blockSize, blockSize);
  
  // 主體
  fill(blockColor);
  stroke(255);
  strokeWeight(2);
  rect(blockX - blockSize/2, springY - blockSize/2, 
       blockSize, blockSize);
  
  // 中心點
  fill(255);
  noStroke();
  ellipse(blockX, springY, 10, 10);
}

// 繪製波形圖
void drawWaveform() {
  // 背景
  fill(255, 255, 255, 30);
  noStroke();
  rect(0, height - 150, width, 150);
  
  // 中心線
  stroke(200);
  strokeWeight(1);
  line(0, height - 75, width, height - 75);
  
  // 波形
  stroke(waveColor);
  strokeWeight(2);
  noFill();
  
  beginShape();
  for (int i = 0; i < waveHistory.size(); i++) {
    float x = map(i, 0, waveHistory.size() - 1, 0, width);
    float y = height - 75 - waveHistory.get(i) * 0.5;
    vertex(x, y);
  }
  endShape();
  
  // 標籤
  fill(255);
  textAlign(LEFT);
  textSize(12);
  text("位移 vs 時間", 10, height - 130);
}

// 資訊面板
void drawInfoPanel() {
  fill(0, 0, 0, 150);
  noStroke();
  rect(10, 10, 260, 180, 5);
  
  fill(255);
  textAlign(LEFT, TOP);
  textSize(14);
  text("彈簧伸縮結構", 25, 25);
  
  textSize(11);
  fill(200);
  text("牆壁→直線→彈簧→直線→方塊", 25, 45);
  
  fill(255);
  textSize(12);
  int y = 70;
  int lineHeight = 20;
  
  float currentAmp = amplitude * pow(damping, time / 10.0);
  text("振幅: " + nf(currentAmp, 0, 1) + " px", 25, y);
  y += lineHeight;
  
  text("頻率: " + nf(frequency, 0, 3) + " Hz", 25, y);
  y += lineHeight;
  
  text("阻尼: " + (damping < 0.999 ? "開啟" : "關閉"), 25, y);
  y += lineHeight;
  
  text("時間: " + nf(time/30, 0, 1) + " s", 25, y);
  y += lineHeight;
  
  text(isPaused ? "狀態: 暫停" : "狀態: 運行", 25, y);
  
  textSize(10);
  fill(200);
  text("按 H 查看幫助", 25, 170);
}

void resetAnimation() {
  time = 0;
  waveHistory.clear();
  for (int i = 0; i < maxHistoryLength; i++) {
    waveHistory.add(0.0);
  }
  println("✓ 動畫已重置");
}

void showHelp() {
  println("\n╔════════════════════════════════╗");
  println("║    彈簧伸縮 - 控制說明        ║");
  println("╠════════════════════════════════╣");
  println("║ 結構：直線→彈簧→直線→方塊   ║");
  println("╠════════════════════════════════╣");
  println("║ R      : 重置動畫              ║");
  println("║ D      : 切換阻尼              ║");
  println("║ +/-    : 調整頻率              ║");
  println("║ 空格   : 暫停/繼續             ║");
  println("║ I      : 資訊面板              ║");
  println("║ W      : 波形圖                ║");
  println("║ A/Z    : 調整振幅              ║");
  println("║ H      : 顯示幫助              ║");
  println("╚════════════════════════════════╝\n");
}

void keyPressed() {
  if (key == 'r' || key == 'R') {
    resetAnimation();
    
  } else if (key == 'd' || key == 'D') {
    if (damping < 0.999) {
      damping = 1.0;
      println("✓ 阻尼關閉");
    } else {
      damping = 0.998;
      println("✓ 阻尼開啟");
    }
    
  } else if (key == '+' || key == '=') {
    frequency = min(0.1, frequency + 0.005);
    println("頻率: " + nf(frequency, 0, 3));
    
  } else if (key == '-' || key == '_') {
    frequency = max(0.01, frequency - 0.005);
    println("頻率: " + nf(frequency, 0, 3));
    
  } else if (key == ' ') {
    isPaused = !isPaused;
    println(isPaused ? "⏸ 暫停" : "▶ 繼續");
    
  } else if (key == 'i' || key == 'I') {
    showInfo = !showInfo;
    
  } else if (key == 'w' || key == 'W') {
    showWave = !showWave;
    
  } else if (key == 'a' || key == 'A') {
    amplitude = min(200, amplitude + 10);
    println("振幅: " + amplitude);
    
  } else if (key == 'z' || key == 'Z') {
    amplitude = max(30, amplitude - 10);
    println("振幅: " + amplitude);
    
  } else if (key == 'h' || key == 'H') {
    showHelp();
  }
}

void mousePressed() {
  showHelp();
}
