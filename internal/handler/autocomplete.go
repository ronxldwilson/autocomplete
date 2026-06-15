package handler

import (
	"encoding/json"
	"net/http"
	"strings"
	"sync"
	"unicode"

	"autocomplete/internal/llm"
)

type AutocompleteRequest struct {
	Text           string `json:"text"`
	MaxTokens      int    `json:"max_tokens,omitempty"`
	NumSuggestions int    `json:"num_suggestions,omitempty"`
}

type Suggestion struct {
	Completion string  `json:"completion"`
	FullText   string  `json:"full_text"`
	LatencyMS  float64 `json:"latency_ms"`
}

type AutocompleteResponse struct {
	Suggestions []Suggestion `json:"suggestions"`
}

func (h *Handler) Autocomplete(w http.ResponseWriter, r *http.Request) {
	var req AutocompleteRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		writeError(w, http.StatusBadRequest, "invalid JSON body")
		return
	}

	text := strings.TrimSpace(req.Text)
	if text == "" {
		writeError(w, http.StatusBadRequest, "text is required")
		return
	}
	if len(text) > 500 {
		writeError(w, http.StatusBadRequest, "text exceeds 500 character limit")
		return
	}

	// Require at least one complete meaningful word
	if !strings.Contains(text, " ") || !hasContentWord(text) {
		writeJSON(w, http.StatusOK, AutocompleteResponse{Suggestions: []Suggestion{}})
		return
	}

	maxTokens := req.MaxTokens
	if maxTokens <= 0 || maxTokens > 20 {
		maxTokens = h.cfg.MaxTokens
	}

	numSuggestions := req.NumSuggestions
	if numSuggestions <= 0 || numSuggestions > 10 {
		numSuggestions = h.cfg.NumSuggestions
	}

	// Determine if we need word-boundary splitting
	promptFull := text
	promptSplit := ""
	partialWord := ""
	needsSplit := false

	if !strings.HasSuffix(req.Text, " ") {
		lastSpace := strings.LastIndex(text, " ")
		if lastSpace >= 0 {
			trailing := text[lastSpace+1:]
			if len(trailing) <= 3 {
				promptSplit = text[:lastSpace]
				partialWord = strings.ToLower(trailing)
				needsSplit = true
			}
		}
	}

	type result struct {
		suggestion Suggestion
		ok         bool
	}

	totalRequests := numSuggestions
	if needsSplit {
		totalRequests = numSuggestions * 2
	}
	if totalRequests > 10 {
		totalRequests = 10
	}

	var wg sync.WaitGroup
	results := make([]result, totalRequests)

	for i := range totalRequests {
		wg.Add(1)
		go func(idx int) {
			defer wg.Done()

			// Alternate between full-text and split strategies
			prompt := promptFull
			filtering := false
			if needsSplit && idx%2 == 0 {
				prompt = promptSplit
				filtering = true
			}

			resp, err := h.client.Complete(llm.CompletionRequest{
				Prompt:        h.cfg.PromptPrefix + prompt,
				NPredict:      maxTokens,
				Temperature:   h.cfg.Temperature,
				TopK:          h.cfg.TopK,
				TopP:          h.cfg.TopP,
				RepeatPenalty: 1.3,
				Stop:          []string{"\n", ".", "?", "!"},
			})
			if err != nil {
				return
			}
			completion := strings.TrimSpace(resp.Content)
			if completion == "" {
				return
			}
			if filtering && !strings.HasPrefix(strings.ToLower(completion), partialWord) {
				return
			}
			if !isCleanCompletion(completion) {
				return
			}
			fullText := prompt + " " + completion
			results[idx] = result{
				suggestion: Suggestion{
					Completion: completion,
					FullText:   fullText,
					LatencyMS:  resp.Timings.PromptMS + resp.Timings.PredictedMS,
				},
				ok: true,
			}
		}(i)
	}
	wg.Wait()

	seen := make(map[string]struct{})
	var suggestions []Suggestion
	for _, r := range results {
		if !r.ok {
			continue
		}
		key := strings.ToLower(r.suggestion.FullText)
		if _, dup := seen[key]; dup {
			continue
		}
		seen[key] = struct{}{}
		suggestions = append(suggestions, r.suggestion)
		if len(suggestions) >= numSuggestions {
			break
		}
	}

	if suggestions == nil {
		suggestions = []Suggestion{}
	}

	writeJSON(w, http.StatusOK, AutocompleteResponse{Suggestions: suggestions})
}

var stopWords = map[string]struct{}{
	"a": {}, "an": {}, "the": {}, "of": {}, "in": {}, "on": {}, "at": {},
	"to": {}, "for": {}, "and": {}, "or": {}, "is": {}, "it": {}, "by": {},
	"with": {}, "from": {}, "as": {}, "be": {}, "do": {}, "so": {}, "if": {},
	"my": {}, "me": {}, "i": {}, "we": {}, "us": {}, "he": {}, "she": {},
	"not": {}, "no": {}, "but": {}, "this": {}, "that": {}, "what": {},
	"how": {}, "who": {}, "which": {}, "where": {}, "when": {}, "why": {},
	"can": {}, "will": {}, "would": {}, "should": {}, "could": {},
	"am": {}, "are": {}, "was": {}, "were": {}, "been": {}, "being": {},
	"have": {}, "has": {}, "had": {}, "does": {}, "did": {},
	"get": {}, "got": {}, "give": {}, "help": {}, "need": {}, "want": {},
}

func hasContentWord(text string) bool {
	words := strings.Fields(strings.ToLower(text))
	for _, w := range words {
		if _, ok := stopWords[w]; !ok {
			return true
		}
	}
	return false
}

var hindiWords = map[string]struct{}{
	"ka": {}, "ke": {}, "ki": {}, "ko": {}, "kya": {}, "hai": {}, "hain": {},
	"mein": {}, "se": {}, "ya": {}, "konsa": {}, "chahiye": {}, "chahta": {},
	"kaise": {}, "kahan": {}, "kaisa": {}, "karna": {}, "karein": {}, "kare": {},
	"dikhao": {}, "batao": {}, "dhundh": {}, "liye": {}, "raha": {}, "hoon": {},
	"hun": {}, "nahi": {}, "bhi": {}, "aur": {}, "pe": {}, "par": {},
	"wala": {}, "wale": {}, "abhi": {}, "bahut": {}, "koi": {}, "loon": {},
	"milta": {}, "milte": {}, "milti": {}, "doon": {}, "hoga": {}, "broadly": {},
}

func isCleanCompletion(s string) bool {
	if len(s) < 2 {
		return false
	}
	ascii := 0
	total := 0
	for _, r := range s {
		total++
		if r < 128 {
			ascii++
		}
	}
	if total > 0 && float64(ascii)/float64(total) < 0.8 {
		return false
	}
	lastChar := rune(s[len(s)-1])
	if unicode.IsLetter(lastChar) {
		words := strings.Fields(s)
		lastWord := words[len(words)-1]
		if len(lastWord) == 1 && unicode.IsUpper(rune(lastWord[0])) {
			return false
		}
	}
	if strings.Count(s, "(") != strings.Count(s, ")") {
		return false
	}
	words := strings.Fields(strings.ToLower(s))
	hindiCount := 0
	for _, w := range words {
		if _, ok := hindiWords[w]; ok {
			hindiCount++
		}
	}
	if len(words) > 0 && float64(hindiCount)/float64(len(words)) > 0.2 {
		return false
	}
	return true
}
