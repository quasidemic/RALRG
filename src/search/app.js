(function () {
    "use strict";

    const ALL_FIELDS = "__all_fields__";
    const MAX_INITIAL_RESULTS = 250;

    const titleFields = [
        "finding_summary",
        "theory_or_framework",
        "method_or_model",
        "study_reference",
        "main_topic",
        "core_concept",
        "paper_filename"
    ];

    const preferredFieldOrder = [
        "paper_filename",
        "finding_summary",
        "theory_or_framework",
        "core_concept",
        "concept_definition",
        "method_or_model",
        "research_design",
        "study_reference",
        "main_topic",
        "main_findings",
        "evidence_or_result",
        "interpretation",
        "qualification_or_boundary_condition",
        "relationship_to_theory",
        "relationship_to_prior_research",
        "population_or_group",
        "population_and_sample",
        "geographic_context",
        "temporal_context",
        "temporal_scope",
        "supporting_quotes",
        "supporting_chunk_ids",
        "citations_used",
        "used_for"
    ];

    const state = {
        records: [],
        fields: [],
        files: [],
        errors: []
    };

    const elements = {
        loadView: document.querySelector("#load-view"),
        searchView: document.querySelector("#search-view"),
        fileInput: document.querySelector("#record-files"),
        reloadFileInput: document.querySelector("#reload-record-files"),
        loadOtherFiles: document.querySelector("#load-other-files"),
        loadStatus: document.querySelector("#load-status"),
        recordCount: document.querySelector("#record-count"),
        fieldCount: document.querySelector("#field-count"),
        fileCount: document.querySelector("#file-count"),
        fileList: document.querySelector("#file-list"),
        searchForm: document.querySelector("#search-form"),
        searchQuery: document.querySelector("#search-query"),
        fieldSelect: document.querySelector("#field-select"),
        resultLimit: document.querySelector("#result-limit"),
        caseSensitive: document.querySelector("#case-sensitive"),
        clearSearch: document.querySelector("#clear-search"),
        resultsTitle: document.querySelector("#results-title"),
        resultsMeta: document.querySelector("#results-meta"),
        results: document.querySelector("#results")
    };

    elements.fileInput.addEventListener("change", async (event) => {
        await loadFiles(Array.from(event.target.files || []));
    });

    elements.reloadFileInput.addEventListener("change", async (event) => {
        await loadFiles(Array.from(event.target.files || []));
    });

    elements.loadOtherFiles.addEventListener("click", () => {
        elements.reloadFileInput.value = "";
        elements.reloadFileInput.click();
    });

    elements.searchForm.addEventListener("submit", (event) => {
        event.preventDefault();
        runSearch();
    });

    elements.searchQuery.addEventListener("input", () => {
        runSearch();
    });

    elements.fieldSelect.addEventListener("change", () => {
        runSearch();
    });

    elements.resultLimit.addEventListener("change", () => {
        runSearch();
    });

    elements.caseSensitive.addEventListener("change", () => {
        runSearch();
    });

    elements.clearSearch.addEventListener("click", () => {
        elements.searchQuery.value = "";
        elements.fieldSelect.value = ALL_FIELDS;
        elements.caseSensitive.checked = false;
        runSearch();
        elements.searchQuery.focus();
    });

    async function loadFiles(files) {
        if (!files.length) {
            return;
        }

        setLoadStatus("Reading selected files...");

        const records = [];
        const fileSummaries = [];
        const errors = [];

        for (const file of files) {
            try {
                const parsed = await parseRecordFile(file);
                records.push(...parsed.records);
                fileSummaries.push({
                    name: file.name,
                    size: file.size,
                    count: parsed.records.length
                });
                errors.push(...parsed.errors);
            } catch (error) {
                errors.push(`${file.name}: ${error.message}`);
            }
        }

        if (!records.length) {
            reportLoadFailure("No JSON object records were found in the selected file(s).");
            return;
        }

        state.records = records;
        state.files = fileSummaries;
        state.fields = collectFields(records);
        state.errors = errors;

        populateFieldSelect(state.fields);
        renderLoadedSummary();
        showSearchView();
        renderResults(records, {
            query: "",
            selectedField: ALL_FIELDS,
            totalMatches: records.length,
            limit: getResultLimit(),
            errors: state.errors
        });
        elements.fileInput.value = "";
        elements.reloadFileInput.value = "";
        elements.searchQuery.focus();
    }

    async function parseRecordFile(file) {
        const text = await file.text();
        const trimmed = text.trim();
        const errors = [];

        if (!trimmed) {
            return { records: [], errors: [`${file.name}: file is empty.`] };
        }

        if (trimmed.startsWith("[")) {
            return parseJsonArrayFile(file, trimmed);
        }

        const records = [];
        const lines = text.split(/\r?\n/);

        lines.forEach((line, index) => {
            const lineNumber = index + 1;
            const cleanLine = line.trim();
            if (!cleanLine) {
                return;
            }

            try {
                const record = JSON.parse(cleanLine);
                if (!record || typeof record !== "object" || Array.isArray(record)) {
                    errors.push(`${file.name}:${lineNumber} is not a JSON object.`);
                    return;
                }
                records.push(addRecordMeta(record, file.name, lineNumber));
            } catch (error) {
                errors.push(`${file.name}:${lineNumber} could not be parsed as JSON.`);
            }
        });

        return { records, errors };
    }

    function parseJsonArrayFile(file, text) {
        const parsed = JSON.parse(text);
        if (!Array.isArray(parsed)) {
            throw new Error("JSON file must contain an array of record objects.");
        }

        const records = [];
        const errors = [];

        parsed.forEach((record, index) => {
            if (!record || typeof record !== "object" || Array.isArray(record)) {
                errors.push(`${file.name}: item ${index + 1} is not a JSON object.`);
                return;
            }
            records.push(addRecordMeta(record, file.name, index + 1));
        });

        return { records, errors };
    }

    function addRecordMeta(record, fileName, lineNumber) {
        return {
            record,
            sourceFile: fileName,
            lineNumber
        };
    }

    function collectFields(records) {
        const fields = new Set();
        records.forEach(({ record }) => {
            Object.keys(record).forEach((field) => fields.add(field));
        });

        return Array.from(fields).sort((a, b) => {
            const aIndex = preferredFieldOrder.indexOf(a);
            const bIndex = preferredFieldOrder.indexOf(b);

            if (aIndex !== -1 || bIndex !== -1) {
                if (aIndex === -1) return 1;
                if (bIndex === -1) return -1;
                return aIndex - bIndex;
            }

            return humanizeField(a).localeCompare(humanizeField(b));
        });
    }

    function populateFieldSelect(fields) {
        elements.fieldSelect.replaceChildren();

        const allOption = document.createElement("option");
        allOption.value = ALL_FIELDS;
        allOption.textContent = "All fields";
        elements.fieldSelect.append(allOption);

        fields.forEach((field) => {
            const option = document.createElement("option");
            option.value = field;
            option.textContent = humanizeField(field);
            elements.fieldSelect.append(option);
        });
    }

    function renderLoadedSummary() {
        elements.recordCount.textContent = String(state.records.length);
        elements.fieldCount.textContent = String(state.fields.length);
        elements.fileCount.textContent = String(state.files.length);
        elements.fileList.replaceChildren();

        state.files.forEach((file) => {
            const chip = document.createElement("span");
            chip.className = "chip";
            chip.textContent = `${file.name} (${file.count})`;
            elements.fileList.append(chip);
        });
    }

    function showSearchView() {
        elements.loadView.classList.add("hidden");
        elements.searchView.classList.remove("hidden");
        setLoadStatus("");
    }

    function runSearch() {
        const query = elements.searchQuery.value.trim();
        const selectedField = elements.fieldSelect.value || ALL_FIELDS;
        const caseSensitive = elements.caseSensitive.checked;
        const limit = getResultLimit();

        const matches = query
            ? state.records.filter(({ record }) => recordMatches(record, query, selectedField, caseSensitive))
            : state.records;

        renderResults(matches, {
            query,
            selectedField,
            totalMatches: matches.length,
            limit,
            errors: state.errors
        });
    }

    function recordMatches(record, query, selectedField, caseSensitive) {
        if (selectedField === ALL_FIELDS) {
            return Object.values(record).some((value) => valueMatches(value, query, caseSensitive));
        }

        return valueMatches(record[selectedField], query, caseSensitive);
    }

    function valueMatches(value, query, caseSensitive) {
        const haystack = valueToSearchText(value);
        if (!haystack) {
            return false;
        }

        if (caseSensitive) {
            return haystack.includes(query);
        }

        return haystack.toLowerCase().includes(query.toLowerCase());
    }

    function valueToSearchText(value) {
        if (value === null || value === undefined) {
            return "";
        }

        if (Array.isArray(value)) {
            return value.map(valueToSearchText).join("\n");
        }

        if (typeof value === "object") {
            return Object.values(value).map(valueToSearchText).join("\n");
        }

        return String(value);
    }

    function renderResults(matches, options) {
        const { query, selectedField, totalMatches, limit, errors } = options;
        const shown = matches.slice(0, limit);
        const fieldLabel = selectedField === ALL_FIELDS ? "all fields" : humanizeField(selectedField);

        elements.results.replaceChildren();
        elements.resultsTitle.textContent = query ? "Search Results" : "Records";
        elements.resultsMeta.textContent = buildResultsMeta(query, fieldLabel, shown.length, totalMatches, limit, errors);

        if (errors && errors.length) {
            elements.results.append(renderErrorSummary(errors));
        }

        if (!shown.length) {
            const empty = document.createElement("div");
            empty.className = "empty-state";
            empty.textContent = query
                ? "No records matched the current search."
                : "No records to display.";
            elements.results.append(empty);
            return;
        }

        shown.forEach((entry, index) => {
            elements.results.append(renderRecordCard(entry, {
                index,
                query,
                selectedField,
                caseSensitive: elements.caseSensitive.checked
            }));
        });
    }

    function buildResultsMeta(query, fieldLabel, shown, totalMatches, limit, errors) {
        const errorSuffix = errors && errors.length
            ? ` ${errors.length} parse warning${errors.length === 1 ? "" : "s"}.`
            : "";

        if (!query) {
            const limitText = totalMatches > shown ? ` Showing first ${shown} of ${totalMatches}.` : "";
            return `Showing loaded records.${limitText}${errorSuffix}`;
        }

        const limitText = totalMatches > limit ? ` Showing first ${shown}.` : "";
        return `${totalMatches} match${totalMatches === 1 ? "" : "es"} in ${fieldLabel}.${limitText}${errorSuffix}`;
    }

    function renderErrorSummary(errors) {
        const details = document.createElement("details");
        details.className = "empty-state";

        const summary = document.createElement("summary");
        summary.textContent = `${errors.length} parse warning${errors.length === 1 ? "" : "s"}`;
        details.append(summary);

        const list = document.createElement("ul");
        list.className = "field-list";
        errors.slice(0, 50).forEach((message) => {
            const item = document.createElement("li");
            item.textContent = message;
            list.append(item);
        });

        if (errors.length > 50) {
            const item = document.createElement("li");
            item.textContent = `And ${errors.length - 50} more.`;
            list.append(item);
        }

        details.append(list);
        return details;
    }

    function renderRecordCard(entry, options) {
        const card = document.createElement("article");
        card.className = "record-card";

        const meta = document.createElement("div");
        meta.className = "record-meta";
        appendChip(meta, `Record ${options.index + 1}`);
        appendChip(meta, entry.sourceFile);
        appendChip(meta, `Line ${entry.lineNumber}`);

        const chunkIds = normalizeToArray(entry.record.supporting_chunk_ids);
        if (chunkIds.length) {
            appendChip(meta, `Chunks ${chunkIds.join(", ")}`);
        }

        card.append(meta);

        const title = document.createElement("h3");
        appendHighlightedText(title, getRecordTitle(entry.record), options.query, options.caseSensitive);
        card.append(title);

        const matchingFields = options.query
            ? getMatchingFields(entry.record, options.query, options.selectedField, options.caseSensitive)
            : [];

        if (matchingFields.length) {
            const matchedMeta = document.createElement("div");
            matchedMeta.className = "record-meta";
            appendChip(matchedMeta, `Matched in ${matchingFields.map(humanizeField).join(", ")}`);
            card.append(matchedMeta);
        }

        const fields = document.createElement("div");
        fields.className = "record-fields";
        getRenderableFields(entry.record).forEach(([field, value]) => {
            fields.append(renderField(field, value, options.query, options.caseSensitive));
        });
        card.append(fields);

        return card;
    }

    function appendChip(parent, text) {
        const chip = document.createElement("span");
        chip.className = "chip";
        chip.textContent = text;
        parent.append(chip);
    }

    function getRecordTitle(record) {
        for (const field of titleFields) {
            if (!isEmptyValue(record[field])) {
                const text = valueToSearchText(record[field]).trim();
                if (text) {
                    return truncateText(text, 240);
                }
            }
        }

        return "Untitled record";
    }

    function getMatchingFields(record, query, selectedField, caseSensitive) {
        const fields = selectedField === ALL_FIELDS ? Object.keys(record) : [selectedField];
        return fields.filter((field) => valueMatches(record[field], query, caseSensitive));
    }

    function getRenderableFields(record) {
        const ordered = [];
        const seen = new Set();

        preferredFieldOrder.forEach((field) => {
            if (Object.prototype.hasOwnProperty.call(record, field) && !isEmptyValue(record[field])) {
                ordered.push([field, record[field]]);
                seen.add(field);
            }
        });

        Object.keys(record)
            .sort((a, b) => humanizeField(a).localeCompare(humanizeField(b)))
            .forEach((field) => {
                if (!seen.has(field) && !isEmptyValue(record[field])) {
                    ordered.push([field, record[field]]);
                }
            });

        return ordered;
    }

    function renderField(field, value, query, caseSensitive) {
        const wrapper = document.createElement("section");
        wrapper.className = "record-field";

        const label = document.createElement("div");
        label.className = "field-name";
        label.textContent = humanizeField(field);
        wrapper.append(label);

        if (field === "supporting_quotes" && Array.isArray(value)) {
            wrapper.append(renderQuoteList(value, query, caseSensitive));
        } else if (Array.isArray(value)) {
            wrapper.append(renderArrayValue(value, query, caseSensitive));
        } else if (value && typeof value === "object") {
            wrapper.append(renderJsonValue(value, query, caseSensitive));
        } else {
            const fieldValue = document.createElement("div");
            fieldValue.className = "field-value";
            appendHighlightedText(fieldValue, String(value), query, caseSensitive);
            wrapper.append(fieldValue);
        }

        return wrapper;
    }

    function renderQuoteList(quotes, query, caseSensitive) {
        const container = document.createElement("div");
        container.className = "quote-list";

        quotes.forEach((quote) => {
            const blockquote = document.createElement("blockquote");
            appendHighlightedText(blockquote, String(quote), query, caseSensitive);
            container.append(blockquote);
        });

        return container;
    }

    function renderArrayValue(values, query, caseSensitive) {
        const scalarValues = values.every((value) => value === null || typeof value !== "object");

        if (scalarValues && values.length <= 12 && values.every((value) => String(value).length <= 80)) {
            const chips = document.createElement("div");
            chips.className = "record-meta";
            values.forEach((value) => appendChip(chips, String(value)));
            return chips;
        }

        const list = document.createElement("ul");
        list.className = "field-list";
        values.forEach((value) => {
            const item = document.createElement("li");
            if (value && typeof value === "object") {
                item.append(renderJsonValue(value, query, caseSensitive));
            } else {
                appendHighlightedText(item, String(value), query, caseSensitive);
            }
            list.append(item);
        });

        return list;
    }

    function renderJsonValue(value, query, caseSensitive) {
        const pre = document.createElement("pre");
        pre.className = "json-value";
        appendHighlightedText(pre, JSON.stringify(value, null, 2), query, caseSensitive);
        return pre;
    }

    function appendHighlightedText(parent, text, query, caseSensitive) {
        if (!query) {
            parent.append(document.createTextNode(text));
            return;
        }

        const source = String(text);
        const needle = caseSensitive ? query : query.toLowerCase();
        const haystack = caseSensitive ? source : source.toLowerCase();
        let cursor = 0;
        let matchIndex = haystack.indexOf(needle, cursor);

        while (matchIndex !== -1) {
            if (matchIndex > cursor) {
                parent.append(document.createTextNode(source.slice(cursor, matchIndex)));
            }

            const mark = document.createElement("mark");
            mark.textContent = source.slice(matchIndex, matchIndex + query.length);
            parent.append(mark);

            cursor = matchIndex + query.length;
            matchIndex = haystack.indexOf(needle, cursor);
        }

        if (cursor < source.length) {
            parent.append(document.createTextNode(source.slice(cursor)));
        }
    }

    function normalizeToArray(value) {
        if (Array.isArray(value)) {
            return value.filter((item) => !isEmptyValue(item)).map(String);
        }

        if (isEmptyValue(value)) {
            return [];
        }

        return [String(value)];
    }

    function isEmptyValue(value) {
        if (value === null || value === undefined) {
            return true;
        }

        if (typeof value === "string") {
            return value.trim() === "";
        }

        if (Array.isArray(value)) {
            return value.length === 0 || value.every(isEmptyValue);
        }

        if (typeof value === "object") {
            return Object.keys(value).length === 0;
        }

        return false;
    }

    function getResultLimit() {
        const value = Number.parseInt(elements.resultLimit.value, 10);
        if (!Number.isFinite(value) || value <= 0) {
            elements.resultLimit.value = String(MAX_INITIAL_RESULTS);
            return MAX_INITIAL_RESULTS;
        }

        return value;
    }

    function humanizeField(field) {
        return String(field)
            .replace(/_/g, " ")
            .replace(/\b\w/g, (letter) => letter.toUpperCase());
    }

    function truncateText(text, maxLength) {
        if (text.length <= maxLength) {
            return text;
        }

        return `${text.slice(0, maxLength - 1)}...`;
    }

    function setLoadStatus(message, isError) {
        elements.loadStatus.textContent = message;
        elements.loadStatus.classList.toggle("error", Boolean(isError));
    }

    function reportLoadFailure(message) {
        setLoadStatus(message, true);

        if (elements.searchView.classList.contains("hidden")) {
            return;
        }

        elements.resultsTitle.textContent = "Files Not Loaded";
        elements.resultsMeta.textContent = message;
        elements.results.replaceChildren();

        const empty = document.createElement("div");
        empty.className = "empty-state";
        empty.textContent = message;
        elements.results.append(empty);
    }
})();
