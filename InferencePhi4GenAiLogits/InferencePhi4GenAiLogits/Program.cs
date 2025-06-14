using Microsoft.ML.OnnxRuntimeGenAI;
using System;
using System.Linq;
using System.Collections.Generic;

// Phi4
var modelPath = @"C:\repos\Phi-4-mini-instruct-onnx\cpu_and_mobile\cpu-int4-rtn-block-32-acc-level-4\";

var systemPrompt =
    "Judge whether the JobDescription meets the requirements based on the Vacancy. Note that the answer can only be \"yes\" or \"no\".";

// initialize model
using var model = new Model(modelPath);
using var tokenizer = new Tokenizer(model);

// Get token IDs for "yes" and "no" (similar to Python code)
var yesTokens = tokenizer.Encode("Yes");
var noTokens = tokenizer.Encode("No");

var yesTokens2 = tokenizer.Encode(" yes");
var noTokens2 = tokenizer.Encode(" no");

// Extract single token IDs (assuming single tokens)
var yesTokenId = yesTokens[0][0]; // First sequence, first token
var noTokenId = noTokens[0][0];   // First sequence, first token

var yesTokenId2 = yesTokens2[0][0]; // First sequence, first token
var noTokenId2 = noTokens2[0][0];   // First sequence, first token

Console.WriteLine($"Yes token ID: {yesTokenId}");
Console.WriteLine($"No token ID: {noTokenId}");
Console.WriteLine($"Yes token ID (with space): {yesTokenId2}");
Console.WriteLine($"No token ID (with space): {noTokenId2}");

// Method to calculate yes/no probabilities from logits
float CalculateYesProbability(Tensor logitsTensor)
{
    // Logits tensor shape is typically [batch_size, sequence_length, vocab_size]
    // For generation, we want the last token's logits: [batch_size, vocab_size]
    // Since we're using single batch, it's [1, vocab_size]
    
    var logitsShape = logitsTensor.Shape();
    Console.WriteLine($"Logits tensor shape: [{string.Join(", ", logitsShape)}]");
    
    // Get the raw logits data as float array using the correct method from the Tensor class
    var logitsData = logitsTensor.GetData<float>().ToArray();
    
    // Calculate the starting index for the last token's logits
    // If shape is [batch, seq_len, vocab_size], we want the last seq_len position
    int vocabSize = (int)logitsShape[^1]; // Last dimension is vocab size
    int lastTokenOffset = 0;
    
    if (logitsShape.Length == 3)
    {
        // Shape: [batch_size, sequence_length, vocab_size]
        int batchSize = (int)logitsShape[0];
        int seqLength = (int)logitsShape[1];
        lastTokenOffset = (seqLength - 1) * vocabSize; // Last token in sequence
    }
    else if (logitsShape.Length == 2)
    {
        // Shape: [batch_size, vocab_size] - already at the right position
        lastTokenOffset = 0;
    }
    
    // Extract logits for "yes" and "no" tokens (both versions)
    float yesLogit = logitsData[lastTokenOffset + yesTokenId];
    float noLogit = logitsData[lastTokenOffset + noTokenId];
    
    float yesLogit2 = logitsData[lastTokenOffset + yesTokenId2];
    float noLogit2 = logitsData[lastTokenOffset + noTokenId2];

    Console.WriteLine($"Yes logit: {yesLogit:F4}, No logit: {noLogit:F4}");
    Console.WriteLine($"Yes logit2: {yesLogit2:F4}, No logit2: {noLogit2:F4}");
    
    // Use the higher probability version (you might want to experiment with this)
    // For now, let's use the version without space
    float maxLogit = Math.Max(yesLogit, noLogit);
    float expYes = (float)Math.Exp(yesLogit - maxLogit);
    float expNo = (float)Math.Exp(noLogit - maxLogit);
    float sumExp = expYes + expNo;
    
    // Probability of "yes"
    float yesProbability = expYes / sumExp;
    
    return yesProbability;
}

// Method to format reranking instruction (similar to Python format_instruction)
string FormatInstruction(string instruction, string query, string document)
{
    if (string.IsNullOrEmpty(instruction))
        instruction = "Given a vacancy title, retrieve relevant job description of the candidate that is suitable for the vacancy";
    
    return $"Vacancy:\"\n {query} \"\n JobDescription:\"\n {document} \"\n";
}

// Method to process a single query-document pair
(float probability, string response) ProcessPair(string query, string document)
{
    // Format the reranking prompt
    string formattedInput = FormatInstruction("", query, document);
    
    // Create the full prompt similar to Qwen format
    var fullPrompt = $"<|system|>{systemPrompt}<|end|><|user|>{formattedInput}<|end|><|assistant|>";
    
    var tokens = tokenizer.Encode(fullPrompt);

    var generatorParams = new GeneratorParams(model);
    generatorParams.SetSearchOption("max_length", 2048);
    generatorParams.SetSearchOption("past_present_share_buffer", false);
    
    using var generator = new Generator(model, generatorParams);
    generator.AppendTokens(tokens[0].ToArray());

    float yesProbability = 0f;
    string response = "";

    // Get logits after processing the prompt (before generation)
    if (!generator.IsDone())
    {
        // Get logits tensor
        // Shape is typically [batch_size, sequence_length, vocab_size] or [batch_size, vocab_size]
        Tensor logitsTensor = generator.GetOutput("logits");
        
        // Calculate yes probability
        yesProbability = CalculateYesProbability(logitsTensor);
        
        // Generate a few tokens to see the actual yes/no response
        using var tokenizerStream = tokenizer.CreateStream();
        
        int tokensGenerated = 0;
        while (!generator.IsDone() && tokensGenerated < 10)
        {
            generator.GenerateNextToken();
            
            var output = tokenizerStream.Decode(generator.GetSequence(0)[^1]);
            response += output;
            tokensGenerated++;
        }
    }
    
    return (yesProbability, response.Trim());
}

// Test data
var queries = new string[]
{
    "What is the capital of China?",
    "Explain gravity",
    "C# Backend developer",
    "C# Backend developer",
    "Unity developer"
};

var documents = new string[]
{
    "A CAT LIKE MILK",
    "Gravity is a force that attracts two bodies towards each other. It gives weight to physical objects and is responsible for the movement of planets around the sun.",
    "Занимался разработкой новых и оптимизацией существующих backend-сервисов для корпоративной системы, обеспечивал интеграцию со сторонними провайдерами данных.¶¶Участвовал в проектировании и разработке сервисов для обновления и миграции устаревших подсистем.¶¶Стек: .NET 6 - 8, EF, MSSQL, PostgreSQL, Swagger",
    "- Руководство группой менеджеров: распределение ресурсов, контроль качества исполняемой работы, менторинг;¶- Выполнение KPI рекламных кампаний;¶- Взаимодействие с аккаунт-менеджерами, backend и frontend отделами, дизайнерами, отделом баинга, консультация аккаунт-менеджеров;¶- Собеседование кандидатов на должность трафик-менеджера;¶- Проведение performance review;¶- Запуск и ведение рекламных кампаний (Senior Traffic Manager);¶- Оптимизация рекламных кампаний по различным верификаторам;¶- Аналитика в Яндекс.Метрика, Google Analytics;¶- Создание отчетов и посткампейн-исследований.¶",
    "Занимался разработкой новых и оптимизацией существующих backend-сервисов для корпоративной системы, обеспечивал интеграцию со сторонними провайдерами данных.¶¶Участвовал в проектировании и разработке сервисов для обновления и миграции устаревших подсистем.¶¶Стек: .NET 6 - 8, EF, MSSQL, PostgreSQL, Swagger"
};

Console.WriteLine("=== BATCH PROCESSING MULTIPLE PAIRS ===");
Console.WriteLine();

// Process all pairs
var results = new List<(string query, string document, float probability, string response)>();

for (int i = 0; i < Math.Min(queries.Length, documents.Length); i++)
{
    Console.WriteLine($"=== Processing Pair {i + 1}/{Math.Min(queries.Length, documents.Length)} ===");
    Console.WriteLine($"Query: {queries[i]}");
    Console.WriteLine($"Document: {documents[i][..Math.Min(100, documents[i].Length)]}...");
    Console.WriteLine();
    
    var (probability, response) = ProcessPair(queries[i], documents[i]);
    results.Add((queries[i], documents[i], probability, response));
    
    Console.WriteLine($"Relevance probability (yes): {probability:F4}");
    Console.WriteLine($"Relevance probability (no): {(1 - probability):F4}");
    Console.WriteLine($"Model response: {response}");
    Console.WriteLine();
    Console.WriteLine("".PadRight(80, '-'));
    Console.WriteLine();
}

// Summary results
Console.WriteLine("=== SUMMARY RESULTS ===");
Console.WriteLine();
for (int i = 0; i < results.Count; i++)
{
    var result = results[i];
    Console.WriteLine($"Pair {i + 1}:");
    Console.WriteLine($"  Query: {result.query}");
    Console.WriteLine($"  Document: {result.document[..Math.Min(50, result.document.Length)]}...");
    Console.WriteLine($"  Yes Probability: {result.probability:F4}");
    Console.WriteLine($"  Model Response: {result.response}");
    Console.WriteLine();
}