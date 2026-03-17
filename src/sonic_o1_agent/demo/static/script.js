/* ============================================================
   Sonic O1 Agent — Frontend Logic
   Handles: video upload, SSE streaming, pipeline progress,
   result rendering (answer, plan, reasoning, reflection)
   ============================================================ */

   const uploadZone   = document.getElementById("upload-zone");
   const videoInput   = document.getElementById("video-input");
   const filePreview  = document.getElementById("file-preview");
   const fileName     = document.getElementById("file-name");
   const fileSize     = document.getElementById("file-size");
   const fileRemove   = document.getElementById("file-remove");
   const queryText    = document.getElementById("query-text");
   const charCountEl  = document.getElementById("char-count");
   const analyzeBtn   = document.getElementById("analyze-btn");
   const pipelineEl   = document.getElementById("pipeline");
   const resultsEl    = document.getElementById("results");

   const MAX_CHARS = 2000;
   let selectedFile = null;
   let currentAbort = null;

   // ============================================================
   // FILE UPLOAD
   // ============================================================

   uploadZone.addEventListener("click", () => videoInput.click());
   uploadZone.addEventListener("dragover", (e) => {
     e.preventDefault();
     uploadZone.classList.add("dragover");
   });
   uploadZone.addEventListener("dragleave", () => {
     uploadZone.classList.remove("dragover");
   });
   uploadZone.addEventListener("drop", (e) => {
     e.preventDefault();
     uploadZone.classList.remove("dragover");
     if (e.dataTransfer.files.length > 0) {
       handleFile(e.dataTransfer.files[0]);
     }
   });

   videoInput.addEventListener("change", () => {
     if (videoInput.files.length > 0) handleFile(videoInput.files[0]);
   });

   fileRemove.addEventListener("click", () => {
     selectedFile = null;
     videoInput.value = "";
     filePreview.classList.add("hidden");
     uploadZone.classList.remove("hidden");
     updateAnalyzeBtn();
   });

   function handleFile(file) {
     if (!file.type.startsWith("video/")) {
       alert("Please select a video file.");
       return;
     }
     selectedFile = file;
     fileName.textContent = file.name;
     fileSize.textContent = formatSize(file.size);
     filePreview.classList.remove("hidden");
     uploadZone.classList.add("hidden");
     updateAnalyzeBtn();
   }

   function formatSize(bytes) {
     if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(0) + " KB";
     return (bytes / (1024 * 1024)).toFixed(1) + " MB";
   }

   // ============================================================
   // EXAMPLE CHIPS
   // ============================================================

   document.querySelectorAll(".example-chip").forEach(chip => {
     chip.addEventListener("click", () => {
       queryText.value = chip.dataset.text;
       queryText.dispatchEvent(new Event("input"));
       queryText.focus();
     });
   });

   // ============================================================
   // CHAR COUNTER + BUTTON STATE
   // ============================================================

   queryText.addEventListener("input", () => {
     const len = queryText.value.length;
     charCountEl.textContent = `${len} / ${MAX_CHARS}`;
     charCountEl.className = "char-count";
     if (len > MAX_CHARS * 0.9) charCountEl.classList.add("warn");
     if (len >= MAX_CHARS)      charCountEl.classList.add("error");
     updateAnalyzeBtn();
   });

   function updateAnalyzeBtn() {
     analyzeBtn.disabled = !(selectedFile && queryText.value.trim().length > 0);
   }

   // ============================================================
   // ANALYZE — SSE STREAM
   // ============================================================

   analyzeBtn.addEventListener("click", runAnalysis);
   queryText.addEventListener("keydown", (e) => {
     if ((e.metaKey || e.ctrlKey) && e.key === "Enter") runAnalysis();
   });

   // Map LangGraph node names to our pipeline UI nodes
   const NODE_MAP = {
     "planning":          "planning",
     "segmentation":      "segmentation",
     "temporal_indexing":  "temporal_indexing",
     "multi_step":        "inference",
     "reasoning":         "inference",
     "direct":            "inference",
     "reflection":        "reflection",
     "cleanup":           "cleanup",
   };

   async function runAnalysis() {
     const query = queryText.value.trim();
     if (!selectedFile || !query) return;
     if (query.length > MAX_CHARS) {
       alert("Query too long.");
       return;
     }

     // Reset UI
     analyzeBtn.disabled = true;
     resultsEl.classList.add("hidden");
     pipelineEl.classList.remove("hidden");
     resetPipeline();

     // Cancel any in-flight request
     if (currentAbort) currentAbort.abort();
     currentAbort = new AbortController();

     // Build form data
     const formData = new FormData();
     formData.append("video", selectedFile);
     formData.append("query", query);

     const startTime = Date.now();
     let completedNodes = new Set();

     // Live elapsed timer on active node
     const timerInterval = setInterval(() => {
       const activeNode = document.querySelector(".pipeline-node.active");
       if (activeNode) {
         const elapsed = ((Date.now() - startTime) / 1000).toFixed(0);
         activeNode.querySelector(".node-time").textContent = elapsed + "s...";
       }
     }, 1000);

     try {
       const res = await fetch("/analyze/stream", {
         method: "POST",
         body: formData,
         signal: currentAbort.signal,
       });

       if (!res.ok) {
         let detail = "Server error (" + res.status + ")";
         try { detail = (await res.json()).detail || detail; } catch {}
         throw new Error(detail);
       }

       const reader = res.body.getReader();
       const decoder = new TextDecoder();
       let buffer = "";

       while (true) {
         const { done, value } = await reader.read();
         if (done) break;

         buffer += decoder.decode(value, { stream: true });
         const lines = buffer.split("\n");
         buffer = lines.pop();

         for (const line of lines) {
           if (!line.startsWith("data: ")) continue;
           let payload;
           try { payload = JSON.parse(line.slice(6)); } catch { continue; }

           if (payload.node !== undefined) {
             // Pipeline progress event
             const uiNode = NODE_MAP[payload.node] || payload.node;
             markNodeDone(uiNode, payload.elapsed);
             completedNodes.add(uiNode);

             // ── PLANNING ──────────────────────────────────────
             if (payload.node === "planning") {
               if (payload.query_type) {
                 addThinking("planning", `Query type: <strong>${escapeHtml(payload.query_type)}</strong>`);
               }
               if (payload.plan) {
                 const p = payload.plan;
                 if (p.duration_seconds) addThinking("planning", `Video duration: <strong>${Math.round(p.duration_seconds)}s</strong> | Max frames: ${p.max_frames}`);
               }
               if (payload.multi_step_plan) {
                 updateNodeDesc("planning", `${payload.multi_step_plan.length} steps planned`);
                 addThinking("planning", `Decomposed into <strong>${payload.multi_step_plan.length} steps:</strong>`);
                 payload.multi_step_plan.forEach((step, i) => {
                   const desc = step.description || step.action || JSON.stringify(step);
                   addThinking("planning", `<strong>${step.step_id || i + 1}.</strong> ${escapeHtml(desc)}`);
                 });
               } else {
                 addThinking("planning", "Direct inference mode — no decomposition needed");
               }
             }

             // ── SEGMENTATION ──────────────────────────────────
             if (payload.node === "segmentation") {
               addThinking("processing", "Video preprocessed for analysis");
             }

             // ── TEMPORAL INDEXING ──────────────────────────────
             if (payload.node === "temporal_indexing") {
               if (payload.temporal_index) {
                 updateNodeDesc("temporal_indexing", `${payload.temporal_segments || "?"} segments indexed`);
                 addThinking("processing", `<strong>Temporal Index</strong> — ${payload.temporal_segments || "?"} segments captioned:`);
                 // Show each segment caption
                 const lines = payload.temporal_index.split("\n").filter(l => l.trim());
                 lines.forEach(line => {
                   addThinking("processing", escapeHtml(line));
                 });
               } else {
                 addThinking("processing", "Temporal index unavailable.");
               }
             }

             // ── INFERENCE (direct / reasoning / multi_step) ───
             if (payload.node === "direct" || payload.node === "reasoning" || payload.node === "multi_step") {
               updateNodeDesc("inference", "Response generated");

               if (payload.reasoning_chain && payload.reasoning_chain.length > 0) {
                 addThinking("inference", `<strong>Chain-of-Thought</strong> — ${payload.reasoning_chain.length} reasoning steps:`);
                 payload.reasoning_chain.forEach(step => {
                   const action = step.action || "";
                   const result = step.result || step.thought || "";
                   addThinking("inference", `<strong>Step ${step.step || "?"}:</strong> ${escapeHtml(action)}`);
                   if (result) addThinking("inference", escapeHtml(result));
                 });
               }

               if (payload.steps_executed) {
                 addThinking("inference", `<strong>Multi-step execution:</strong> ${Object.keys(payload.steps_executed).length} steps completed`);
               }

               if (payload.response) {
                 addThinking("inference", `<strong>Generated Response:</strong>`);
                 addThinkingBlock("inference", payload.response);
               }

               if (payload.evidence) {
                 if (payload.evidence.video) {
                   const v = payload.evidence.video;
                   addThinking("inference", `Evidence: ${v.frames_analyzed} frames analyzed, ${v.coverage_sec?.toFixed(0) || "?"}s coverage`);
                 }
                 if (payload.evidence.audio) {
                   const a = payload.evidence.audio;
                   addThinking("inference", `Evidence: ${a.chunks_analyzed} audio chunks, ${a.coverage_sec?.toFixed(0) || "?"}s coverage`);
                 }
               }
             }

             // ── REFLECTION ────────────────────────────────────
             if (payload.node === "reflection") {
               const ref = payload.reflection || {};
               const conf = ref.final_confidence || ref.confidence;
               if (conf) {
                 updateNodeDesc("reflection", `Confidence: ${(conf * 100).toFixed(0)}%`);
                 addThinking("reflection", `Self-evaluation confidence: <strong>${(conf * 100).toFixed(0)}%</strong>`);
               }
               if (ref.scores) {
                 const s = ref.scores;
                 const parts = [];
                 if (s.completeness != null) parts.push(`Completeness: ${s.completeness}/10`);
                 if (s.accuracy != null) parts.push(`Accuracy: ${s.accuracy}/10`);
                 if (s.clarity != null) parts.push(`Clarity: ${s.clarity}/10`);
                 if (s.evidence != null) parts.push(`Evidence: ${s.evidence}/10`);
                 if (parts.length) addThinking("reflection", parts.join(" · "));
               }
               if (ref.strengths && ref.strengths.length) {
                 const clean = ref.strengths.filter(s => s && s.replace(/\*/g, "").trim().length > 0);
                 if (clean.length) addThinking("reflection", `<strong>Strengths:</strong> ${clean.map(s => escapeHtml(s)).join("; ")}`);
               }
               if (ref.weaknesses && ref.weaknesses.length) {
                 const clean = ref.weaknesses.filter(w => w && w.replace(/\*/g, "").trim().length > 0);
                 if (clean.length) addThinking("reflection", `<strong>Weaknesses:</strong> ${clean.map(w => escapeHtml(w)).join("; ")}`);
               }
               if (ref.total_attempts && ref.total_attempts > 1) {
                 addThinking("reflection", `Refined over <strong>${ref.total_attempts} attempts</strong>`);
               }
               if (payload.hallucination_assessment) {
                 const h = payload.hallucination_assessment;
                 if (h.has_hallucination) {
                   addThinking("reflection", `⚠ Hallucination detected (severity: <strong>${h.severity}</strong>)`);
                 } else {
                   addThinking("reflection", `✓ No hallucinations detected`);
                 }
               }
               if (payload.was_refined) {
                 addThinking("reflection", `Response was <strong>refined</strong> based on self-evaluation`);
               }
             }

             // ── CLEANUP ───────────────────────────────────────
             if (payload.node === "cleanup") {
               addThinking("processing", "✓ Pipeline complete — rendering final results");
             }

             // Activate next node
             const allNodes = ["planning", "segmentation", "temporal_indexing", "inference", "reflection", "cleanup"];
             const idx = allNodes.indexOf(uiNode);
             if (idx >= 0 && idx < allNodes.length - 1) {
               activateNode(allNodes[idx + 1]);
             }

           } else if (payload.result !== undefined) {
             // Final result
             clearInterval(timerInterval);
             renderResults(payload.result);

           } else if (payload.error !== undefined) {
             clearInterval(timerInterval);
             throw new Error(payload.error);
           }
         }
       }

     } catch (err) {
       clearInterval(timerInterval);
       alert("Analysis failed: " + err.message);
     }

     analyzeBtn.disabled = false;
   }

   // ============================================================
   // PIPELINE UI
   // ============================================================

   function updateNodeDesc(nodeName, text) {
     const el = document.querySelector(`.pipeline-node[data-node="${nodeName}"] .node-desc`);
     if (el) el.textContent = text;
   }

   function addThinking(type, html) {
     const log = document.getElementById("thinking-log");
     if (!log) return;
     const entry = document.createElement("div");
     entry.className = "thinking-entry";
     entry.innerHTML = `
       <span class="thinking-badge ${type}">${type}</span>
       <span class="thinking-text">${html}</span>
     `;
     log.appendChild(entry);
     log.scrollTop = log.scrollHeight;
   }

   function addThinkingBlock(type, text) {
     const log = document.getElementById("thinking-log");
     if (!log) return;
     const entry = document.createElement("div");
     entry.className = "thinking-entry";
     entry.innerHTML = `
       <span class="thinking-badge ${type}">${type}</span>
       <div class="thinking-block">${renderMarkdown(text)}</div>
     `;
     log.appendChild(entry);
     log.scrollTop = log.scrollHeight;
   }

   function resetPipeline() {
     document.querySelectorAll(".pipeline-node").forEach(n => {
       n.classList.remove("active", "done");
       n.querySelector(".node-time").textContent = "";
     });
     document.querySelectorAll(".pipeline-connector").forEach(c => {
       c.classList.remove("done");
     });
     // Clear thinking log
     const log = document.getElementById("thinking-log");
     if (log) log.innerHTML = "";
     // Reset node descriptions
     updateNodeDesc("planning", "Decomposing query into steps");
     updateNodeDesc("segmentation", "Splitting video into segments");
     updateNodeDesc("temporal_indexing", "Captioning segments for grounding");
     updateNodeDesc("inference", "Multi-step reasoning & analysis");
     updateNodeDesc("reflection", "Self-evaluation & refinement");
     updateNodeDesc("cleanup", "Finalising results");
     // Activate first node
     activateNode("planning");
   }

   function activateNode(nodeName) {
     const el = document.querySelector(`.pipeline-node[data-node="${nodeName}"]`);
     if (el && !el.classList.contains("done")) {
       el.classList.add("active");
     }
   }

   function markNodeDone(nodeName, elapsed) {
     const el = document.querySelector(`.pipeline-node[data-node="${nodeName}"]`);
     if (!el) return;
     el.classList.remove("active");
     el.classList.add("done");
     if (elapsed != null) {
       el.querySelector(".node-time").textContent = elapsed.toFixed(1) + "s";
     }

     // Mark connector before this node as done
     const prev = el.previousElementSibling;
     if (prev && prev.classList.contains("pipeline-connector")) {
       prev.classList.add("done");
     }
   }

   // ============================================================
   // RENDER RESULTS
   // ============================================================

   function renderResults(data) {
     resultsEl.classList.remove("hidden");
     resultsEl.scrollIntoView({ behavior: "smooth", block: "start" });

     // Answer — render with markdown formatting
     document.getElementById("answer-text").innerHTML = renderMarkdown(data.response || "");

     // Confidence
     renderConfidence(data);

     // Multi-step plan
     renderPlan(data);

     // Reasoning chain
     renderReasoning(data);

     // Reflection
     renderReflection(data);
   }

   function renderConfidence(data) {
     const badge = document.getElementById("confidence-value");
     let confidence = null;

     if (data.reflection && data.reflection.final_confidence != null) {
       confidence = data.reflection.final_confidence;
     } else if (data.reflection && data.reflection.confidence != null) {
       confidence = data.reflection.confidence;
     } else if (data.confidence != null) {
       confidence = data.confidence;
     }

     if (confidence != null) {
       const pct = (confidence * 100).toFixed(0) + "%";
       badge.textContent = pct;
       badge.className = "confidence-value";
       if (confidence >= 0.8)      badge.classList.add("high");
       else if (confidence >= 0.5) badge.classList.add("medium");
       else                        badge.classList.add("low");
     } else {
       badge.textContent = "—";
       badge.className = "confidence-value";
     }
   }

   function renderPlan(data) {
     const section = document.getElementById("plan-section");
     const list = document.getElementById("plan-list");
     list.innerHTML = "";

     const plan = data.multi_step_plan || [];
     if (plan.length === 0) {
       section.classList.add("hidden");
       return;
     }

     section.classList.remove("hidden");

     plan.forEach((step, idx) => {
       const el = document.createElement("div");
       el.className = "plan-step";
       el.innerHTML = `
         <span class="plan-step-num">Step ${step.step_id || idx + 1}</span>
         <span class="plan-step-text">${escapeHtml(step.description || step.action || JSON.stringify(step))}</span>
       `;
       list.appendChild(el);
     });
   }

   function renderReasoning(data) {
     const section = document.getElementById("reasoning-section");
     const list = document.getElementById("reasoning-list");
     list.innerHTML = "";

     const chain = data.reasoning_chain || [];
     if (chain.length === 0) {
       section.classList.add("hidden");
       return;
     }

     section.classList.remove("hidden");

     chain.forEach((step) => {
       const el = document.createElement("div");
       el.className = "reasoning-step";
       el.innerHTML = `
         <div class="reasoning-step-header">
           <span class="reasoning-step-badge">Step ${step.step || "?"}</span>
           <span class="reasoning-step-action">${escapeHtml(step.action || "")}</span>
         </div>
         <div class="reasoning-step-body">${renderMarkdown(step.result || step.thought || "")}</div>
       `;
       list.appendChild(el);
     });
   }

   function renderReflection(data) {
     const section = document.getElementById("reflection-section");
     const ref = data.reflection;

     if (!ref) {
       section.classList.add("hidden");
       return;
     }

     section.classList.remove("hidden");

     // Score cards — only available in single-shot reflection mode
     const scoresEl = document.getElementById("score-cards");
     scoresEl.innerHTML = "";

     const scores = ref.scores || {};
     const criteria = ["completeness", "accuracy", "clarity", "evidence"];
     let hasScores = false;

     criteria.forEach(key => {
       const val = scores[key];
       if (val == null) return;
       hasScores = true;
       const pct = val * 10;
       const tier = pct >= 80 ? "high" : pct >= 50 ? "medium" : "low";

       const card = document.createElement("div");
       card.className = "score-card";
       card.innerHTML = `
         <span class="score-card-label">${key}</span>
         <span class="score-card-value ${tier}">${val}<span style="font-size:14px;font-weight:400;color:var(--text-3)">/10</span></span>
         <div class="score-card-bar"><div class="score-card-fill ${tier}" style="width:${pct}%"></div></div>
       `;
       scoresEl.appendChild(card);
     });

     // In iterative mode, show total attempts as a summary card if no scores
     if (!hasScores && ref.total_attempts) {
       const card = document.createElement("div");
       card.className = "score-card";
       card.innerHTML = `
         <span class="score-card-label">Refinement Rounds</span>
         <span class="score-card-value" style="color:var(--accent)">${ref.total_attempts}</span>
         <div class="score-card-bar"><div class="score-card-fill high" style="width:100%"></div></div>
       `;
       scoresEl.appendChild(card);
     }

     // Hallucination
     const hallCard = document.getElementById("hallucination-card");
     const hallValue = document.getElementById("hallucination-value");
     const hall = data.hallucination_assessment;

     if (hall) {
       hallCard.classList.remove("hidden");
       const sev = (hall.severity || "NONE").toLowerCase();
       hallValue.textContent = hall.has_hallucination ? sev.toUpperCase() : "None detected";
       hallValue.className = "hallucination-value " + (hall.has_hallucination ? sev : "none");
     } else {
       hallCard.classList.add("hidden");
     }

     // Refinement history
     const refSection = document.getElementById("refinement-section");
     const refList = document.getElementById("refinement-list");
     refList.innerHTML = "";

     const history = data.refinement_history || (ref.refinement_history) || [];
     if (history.length > 0) {
       refSection.classList.remove("hidden");
       history.forEach(item => {
         const el = document.createElement("div");
         el.className = "refinement-item";
         const recClass = (item.recommendation || "").toLowerCase();
         el.innerHTML = `
           <span class="refinement-attempt">Attempt ${item.attempt}</span>
           <span class="refinement-confidence">${(item.confidence * 100).toFixed(0)}%</span>
           <span class="refinement-rec ${recClass}">${item.recommendation || ""}</span>
         `;
         refList.appendChild(el);
       });
     } else {
       refSection.classList.add("hidden");
     }
   }

   // ============================================================
   // COPY BUTTON
   // ============================================================

   document.getElementById("copy-btn").addEventListener("click", () => {
     const text = document.getElementById("answer-text").textContent;
     if (!text) return;
     navigator.clipboard.writeText(text).then(() => {
       const btn = document.getElementById("copy-btn");
       const original = btn.innerHTML;
       btn.innerHTML = `<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="20 6 9 17 4 12"/></svg> Copied!`;
       setTimeout(() => { btn.innerHTML = original; }, 1800);
     });
   });

   // ============================================================
   // UTILITY
   // ============================================================

   function escapeHtml(str) {
     return String(str || "")
       .replace(/&/g, "&amp;")
       .replace(/</g, "&lt;")
       .replace(/>/g, "&gt;")
       .replace(/"/g, "&quot;")
       .replace(/\n/g, "<br/>");
   }

   /**
    * Lightweight markdown → HTML renderer.
    * Handles: **bold**, *italic*, `code`, headings (##), bullet lists (* / -),
    * numbered lists (1.), and paragraphs.
    */
   function renderMarkdown(text) {
     if (!text) return "";

     // Escape HTML entities first
     let html = String(text)
       .replace(/&/g, "&amp;")
       .replace(/</g, "&lt;")
       .replace(/>/g, "&gt;");

     // Split into lines for block-level processing
     const lines = html.split("\n");
     const output = [];
     let inList = false;
     let listType = null; // 'ul' or 'ol'

     for (let i = 0; i < lines.length; i++) {
       let line = lines[i];

       // Headings
       if (/^####\s+(.+)/.test(line)) {
         closeList();
         output.push(`<h4 class="md-h4">${RegExp.$1}</h4>`);
         continue;
       }
       if (/^###\s+(.+)/.test(line)) {
         closeList();
         output.push(`<h3 class="md-h3">${RegExp.$1}</h3>`);
         continue;
       }
       if (/^##\s+(.+)/.test(line)) {
         closeList();
         output.push(`<h2 class="md-h2">${RegExp.$1}</h2>`);
         continue;
       }

       // Bullet list (* or -)
       const bulletMatch = line.match(/^\s*[\*\-]\s+(.+)/);
       if (bulletMatch) {
         if (listType !== "ul") { closeList(); output.push("<ul class='md-ul'>"); inList = true; listType = "ul"; }
         output.push(`<li>${inlineFormat(bulletMatch[1])}</li>`);
         continue;
       }

       // Numbered list (1. 2. etc)
       const numMatch = line.match(/^\s*(\d+)\.\s+(.+)/);
       if (numMatch) {
         if (listType !== "ol") { closeList(); output.push("<ol class='md-ol'>"); inList = true; listType = "ol"; }
         output.push(`<li>${inlineFormat(numMatch[2])}</li>`);
         continue;
       }

       // Empty line = paragraph break
       if (line.trim() === "") {
         closeList();
         output.push("<br/>");
         continue;
       }

       // Regular text line
       closeList();
       output.push(`<p class="md-p">${inlineFormat(line)}</p>`);
     }

     closeList();
     return output.join("\n");

     function closeList() {
       if (inList) {
         output.push(listType === "ol" ? "</ol>" : "</ul>");
         inList = false;
         listType = null;
       }
     }
   }

   /** Inline markdown: **bold**, *italic*, `code` */
   function inlineFormat(text) {
     return text
       .replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>")
       .replace(/\*(.+?)\*/g, "<em>$1</em>")
       .replace(/`(.+?)`/g, "<code class='md-code'>$1</code>");
   }
