

## [METHODOLOGY SHIFT]

### 📚 EMNLP 2025 Research Summary: English as Defense Proxy (E-Proxy)

**Topic:** Mitigating Multilingual Jailbreaks in Large Language Models (LLMs).

**🚀 Novel Framework / Algorithm:**
*   **E-Proxy:** A unified safety approach that leverages English, the advantage language of LLMs, as a universal safety anchor.
*   **Mechanism:** Instead of traditional multilingual safety alignment (which is prone to "translationese" and an "alignment tax"), E-Proxy combines **Parametric Safety** with *Translational Safety*. It elicits inherent English safety knowledge using fixed jailbreak prompts in English, then transfers this across target languages via simple language-mapping instructions (e.g., “Please answer in {target language}”).
*   **Data Construction:** The approach uses $[p_l \oplus x_i]$ (English prompt) to transfer knowledge, bypassing the need for costly and error-prone translation of both prompts ($\operatorname{trans}(x_i, l)$) and safe responses.

**🔬 Key Technical Findings & Systemic Changes:**
*   **Safety Anchor Role:** English serves as a critical anchor, suggesting that LLMs integrate core safety thinking in English even when constrained to produce target language outputs (demonstrated by logit analysis using $p_l \oplus x_i$).
*   **Weight Perturbation:** Using English prompts results in significantly **lower weight perturbation** during training compared to translation-based methods, indicating superior preservation of the model’s general capabilities and reducing alignment overhead.

**🔢 Performance Metrics & Benchmarks (MultiJail):**
| Metric | Result / Finding | Significance |
| :--- | :--- | :--- |
| **Jailbreak Defense** | Blocks over 99% of jailbreak attempts. | State-of-the-art safety performance. |
| **Usefulness Retention** | Retains ~95% average task performance. | Demonstrates the trade-off mitigation (low alignment tax). |
| **Language Space Impact** | E/L configuration (English Prompt, Target Language Response) performing best in both Safety and Usefulness. | Confirms optimal method design for robust cross-lingual defense. |



## [METHODOLOGY/ARCHITECTURE INSIGHT]
## Architectural Insight: English as Defense Proxy (E-Proxy) for Multilingual LLM Safety

**Source:** Findings of EMNLP 2025, 'English as Defense Proxy: Mitigating Multilingual Jailbreak via Eliciting English Safety Knowledge'

**Core Problem Addressed:**
Traditional Large Language Model (LLM) safety alignment struggles with multilingual consistency. Attackers exploit the disparity by translating harmful prompts into low-resource languages to bypass safeguards. Existing methods rely on costly full cross-language alignment, incurring a "multilingual alignment tax" and suffering from translation artifacts ("translationese").

**Proposed Solution: E-Proxy Framework**
E-Proxy leverages English (the high-resource language) as a universal safety anchor, fundamentally changing how multilingual safety is taught. Instead of translating entire datasets ($\hat{D}_l$), the model's inherent English safety knowledge ($x_i$) is retained and amplified to be transferable.

**Technical Methodology & Specifications:**
1. **Safety Knowledge Elicitation:** The core process involves using fixed English jailbreak prompts ($x_i$ with pre-pended language mapping instructions $p_l = 	ext{"Please answer in}\{	ext{target language}\}$). This method activates the model's existing, robust English safety knowledge (Parametric Safety) via logit lens analysis.
2. **Safety Anchor Function:** E-Proxy augments traditional "translational safety" with this Parametric Safety anchor, minimizing alignment overhead and ensuring cross-lingual robustness without resource-intensive multilingual fine-tuning.
3. **Design Constraints/Optimal Strategy:** The study identifies that the optimal configuration involves using English for the Prompt (to preserve utility) and enforcing the target language for the Response (to enhance safety).

**Quantitative Results & Performance Guarantees:**
*   **Jailbreak Mitigation:** On the MultiJail benchmark, E-Proxy blocks over 99% of jailbreak attempts.
*   **Utility Retention:** It maintains over 95% average task performance across both English and non-English settings.
*   **Model Stability:** The framework minimizes weight perturbation (quantified by Principal Angle Distance, PAD), indicating significantly better preservation of the model's general capabilities compared to translation-based methods.

**Conclusion for Implementation:** Utilizing a high-resource pivot language like English in the prompt space is critical both for scalable safety knowledge transfer and mitigating the 'multilingual alignment tax.'
