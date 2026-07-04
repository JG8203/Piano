#include <JuceHeader.h>

#include "AudioFFT.h"
#include "qiano.h"

#include <algorithm>
#include <array>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <mutex>
#include <numeric>
#include <optional>
#include <random>
#include <regex>
#include <sstream>
#include <string>
#include <tuple>
#include <unordered_set>
#include <vector>

#include <pagmo/algorithm.hpp>
#include <pagmo/algorithms/pso_gen.hpp>
#include <pagmo/batch_evaluators/thread_bfe.hpp>
#include <pagmo/bfe.hpp>
#include <pagmo/population.hpp>
#include <pagmo/problem.hpp>
#include <pagmo/threading.hpp>
#include <pagmo/types.hpp>

extern Param params[NumParams];

namespace
{
constexpr int kSampleRate = 48000;
constexpr int kBlockSize = 512;
constexpr double kPi = 3.1415926535897932384626433832795;
constexpr float kEpsilon = 1.0e-7f;
constexpr float kInvalidGenomeLoss = 1.0e9f;
constexpr float kFitParameterMin = 0.2f;
constexpr float kFitParameterMax = 0.8f;

struct Options
{
    std::string manifestPath;
    std::string outputPath = "piano_fit_result.json";
    std::string metricsPath;
    std::string evalGenomesPath;
    std::string evalOutputPath;
    std::string fixedGenomePath;
    std::string subset = "pilot";
    std::string exportDir;
    int maxEvaluations = 10000;
    int population = 40;
    double omega = 0.7298;
    double eta1 = 2.05;
    double eta2 = 2.05;
    double maxVelocity = 0.5;
    unsigned psoVariant = 5;
    unsigned neighbourhoodType = 2;
    unsigned neighbourhoodParam = 4;
    unsigned psoVerbosity = 1;
    float maxSeconds = 6.0f;
    uint32_t seed = 12345;
    bool printModelInfo = false;
    bool serveJsonl = false;
    bool psoMemory = false;
    bool parallelEvaluations = true;
};

struct ManifestRecord
{
    std::string path;
    int midiNote = 60;
    char layer = 'M';
    float targetVelocity = 0.65f;
    int sampleRate = kSampleRate;
    int channels = 2;
    float duration = 0.0f;
    float peak = 0.0f;
    float rms = 0.0f;
};

struct MonoSample
{
    std::vector<float> samples;
    int sampleRate = kSampleRate;
};

struct ScaleFeatures
{
    int fftSize = 0;
    int melBands = 64;
    std::vector<float> melFrames;
    std::vector<float> meanLogMel;
    float targetEnergy = 0.0f;
    float centroid = 0.0f;
};

struct AudioFeatures
{
    std::vector<ScaleFeatures> scales;
    std::array<float, 13> mfcc {};
    std::vector<float> envelope;
};

struct TargetExample
{
    ManifestRecord record;
    MonoSample audio;
    AudioFeatures features;
};

std::string jsonEscape(const std::string& value)
{
    std::ostringstream out;
    for (char c : value)
    {
        switch (c)
        {
            case '\\': out << "\\\\"; break;
            case '"': out << "\\\""; break;
            case '\n': out << "\\n"; break;
            case '\r': out << "\\r"; break;
            case '\t': out << "\\t"; break;
            default: out << c; break;
        }
    }
    return out.str();
}

std::string jsonUnescape(std::string value)
{
    std::string out;
    out.reserve(value.size());
    for (size_t i = 0; i < value.size(); ++i)
    {
        if (value[i] == '\\' && i + 1 < value.size())
        {
            char next = value[++i];
            switch (next)
            {
                case 'n': out.push_back('\n'); break;
                case 'r': out.push_back('\r'); break;
                case 't': out.push_back('\t'); break;
                default: out.push_back(next); break;
            }
        }
        else
        {
            out.push_back(value[i]);
        }
    }
    return out;
}

std::optional<std::string> stringField(const std::string& line, const std::string& key)
{
    const std::regex re("\"" + key + "\"\\s*:\\s*\"((?:\\\\.|[^\"])*)\"");
    std::smatch match;
    if (std::regex_search(line, match, re))
        return jsonUnescape(match[1].str());
    return std::nullopt;
}

std::optional<double> numberField(const std::string& line, const std::string& key)
{
    const std::regex re("\"" + key + "\"\\s*:\\s*(-?[0-9]+(?:\\.[0-9]+)?(?:[eE][+-]?[0-9]+)?)");
    std::smatch match;
    if (std::regex_search(line, match, re))
        return std::stod(match[1].str());
    return std::nullopt;
}

void printUsage(std::ostream& out = std::cout)
{
    out << "usage: PianoFit --manifest <manifest.jsonl> [options]\n"
        << "options:\n"
        << "  --output <file>        Result JSON path (default piano_fit_result.json)\n"
        << "  --metrics <file>       Optional JSONL metric stream for W&B/wrappers\n"
        << "  --eval-genomes <file>  Evaluate external genome JSONL instead of running built-in PSO\n"
        << "  --eval-output <file>   Output JSONL losses for --eval-genomes\n"
        << "  --serve-jsonl          Serve JSONL evaluator requests on stdin/stdout\n"
        << "  --fixed-genome <file>  Evaluate/export one external genome JSON/JSONL instead of optimizing\n"
        << "  --subset pilot|all     Dataset subset (default pilot)\n"
        << "  --max-evals <n>        Evaluation budget; 0 evaluates defaults only\n"
        << "  --population <n>       PSO swarm population (default 40)\n"
        << "  --omega <x>            PSO inertia/constriction coefficient (default 0.7298)\n"
        << "  --eta1 <x>             PSO cognitive coefficient (default 2.05)\n"
        << "  --eta2 <x>             PSO social/neighbourhood coefficient (default 2.05)\n"
        << "  --max-vel <x>          Max particle velocity as fraction of bounds (default 0.5)\n"
        << "  --pso-variant <n>      pagmo PSO variant 1..6 (default 5)\n"
        << "  --neighb-type <n>      pagmo PSO topology 1..4 (default 2)\n"
        << "  --neighb-param <n>     pagmo PSO neighbourhood parameter (default 4)\n"
        << "  --pso-verbosity <n>    pagmo log/screen interval in generations (default 1, 0 disables)\n"
        << "  --pso-memory           Keep pagmo PSO memory across evolve calls\n"
        << "  --serial-evals         Disable pagmo thread_bfe parallel batch fitness evaluation\n"
        << "  --sigma <x>            Deprecated; use --max-vel for PSO velocity bounds\n"
        << "  --max-seconds <x>      Target/render crop duration (default 6)\n"
        << "  --export-dir <dir>     Optional target/render WAV export directory\n"
        << "  --seed <n>             RNG seed\n"
        << "  --print-model-info     Print model metadata JSON and exit\n";
}

Options parseOptions(int argc, char** argv)
{
    Options options;
    for (int i = 1; i < argc; ++i)
    {
        const std::string arg = argv[i];
        auto requireValue = [&](const char* name) -> std::string {
            if (i + 1 >= argc)
                throw std::runtime_error(std::string("Missing value for ") + name);
            return argv[++i];
        };

        if (arg == "--manifest")
            options.manifestPath = requireValue("--manifest");
        else if (arg == "--output")
            options.outputPath = requireValue("--output");
        else if (arg == "--metrics")
            options.metricsPath = requireValue("--metrics");
        else if (arg == "--eval-genomes")
            options.evalGenomesPath = requireValue("--eval-genomes");
        else if (arg == "--eval-output")
            options.evalOutputPath = requireValue("--eval-output");
        else if (arg == "--fixed-genome")
            options.fixedGenomePath = requireValue("--fixed-genome");
        else if (arg == "--subset")
            options.subset = requireValue("--subset");
        else if (arg == "--max-evals")
            options.maxEvaluations = std::stoi(requireValue("--max-evals"));
        else if (arg == "--population")
            options.population = std::stoi(requireValue("--population"));
        else if (arg == "--sigma")
            throw std::runtime_error("--sigma is deprecated for pagmo PSO; use --max-vel to control maximum particle velocity");
        else if (arg == "--omega")
            options.omega = std::stod(requireValue("--omega"));
        else if (arg == "--eta1")
            options.eta1 = std::stod(requireValue("--eta1"));
        else if (arg == "--eta2")
            options.eta2 = std::stod(requireValue("--eta2"));
        else if (arg == "--max-vel")
            options.maxVelocity = std::stod(requireValue("--max-vel"));
        else if (arg == "--pso-variant")
            options.psoVariant = static_cast<unsigned>(std::stoul(requireValue("--pso-variant")));
        else if (arg == "--neighb-type")
            options.neighbourhoodType = static_cast<unsigned>(std::stoul(requireValue("--neighb-type")));
        else if (arg == "--neighb-param")
            options.neighbourhoodParam = static_cast<unsigned>(std::stoul(requireValue("--neighb-param")));
        else if (arg == "--pso-verbosity")
            options.psoVerbosity = static_cast<unsigned>(std::stoul(requireValue("--pso-verbosity")));
        else if (arg == "--max-seconds")
            options.maxSeconds = std::stof(requireValue("--max-seconds"));
        else if (arg == "--export-dir")
            options.exportDir = requireValue("--export-dir");
        else if (arg == "--seed")
            options.seed = static_cast<uint32_t>(std::stoul(requireValue("--seed")));
        else if (arg == "--print-model-info")
            options.printModelInfo = true;
        else if (arg == "--serve-jsonl")
            options.serveJsonl = true;
        else if (arg == "--pso-memory")
            options.psoMemory = true;
        else if (arg == "--serial-evals")
            options.parallelEvaluations = false;
        else if (arg == "--help" || arg == "-h")
        {
            printUsage();
            std::exit(0);
        }
        else
        {
            throw std::runtime_error("Unknown argument: " + arg);
        }
    }

    if (options.manifestPath.empty() && ! options.printModelInfo)
        throw std::runtime_error("--manifest is required");
    if (! options.evalGenomesPath.empty() && options.evalOutputPath.empty())
        throw std::runtime_error("--eval-output is required with --eval-genomes");
    if (options.population < 4)
        throw std::runtime_error("--population must be at least 4");
    if (options.omega < 0.0 || options.omega > 1.0)
        throw std::runtime_error("--omega must be in the [0, 1] interval");
    if (options.eta1 < 0.0 || options.eta1 > 4.0)
        throw std::runtime_error("--eta1 must be in the [0, 4] interval");
    if (options.eta2 < 0.0 || options.eta2 > 4.0)
        throw std::runtime_error("--eta2 must be in the [0, 4] interval");
    if (options.maxVelocity <= 0.0 || options.maxVelocity > 1.0)
        throw std::runtime_error("--max-vel must be in the (0, 1] interval");
    if (options.psoVariant < 1 || options.psoVariant > 6)
        throw std::runtime_error("--pso-variant must be one of 1..6");
    if (options.neighbourhoodType < 1 || options.neighbourhoodType > 4)
        throw std::runtime_error("--neighb-type must be one of 1..4");
    if (options.neighbourhoodParam == 0)
        throw std::runtime_error("--neighb-param must be at least 1");
    return options;
}

std::vector<ManifestRecord> readManifest(const std::string& path)
{
    std::ifstream in(path);
    if (! in)
        throw std::runtime_error("Could not open manifest: " + path);

    std::vector<ManifestRecord> records;
    std::string line;
    while (std::getline(in, line))
    {
        if (line.empty())
            continue;

        ManifestRecord record;
        record.path = stringField(line, "path").value_or("");
        auto layer = stringField(line, "layer").value_or("M");
        record.layer = layer.empty() ? 'M' : layer[0];
        record.midiNote = static_cast<int>(numberField(line, "midi_note").value_or(60));
        record.targetVelocity = static_cast<float>(numberField(line, "target_velocity").value_or(0.65));
        record.sampleRate = static_cast<int>(numberField(line, "sample_rate").value_or(kSampleRate));
        record.channels = static_cast<int>(numberField(line, "channels").value_or(2));
        record.duration = static_cast<float>(numberField(line, "duration").value_or(0.0));
        record.peak = static_cast<float>(numberField(line, "peak").value_or(0.0));
        record.rms = static_cast<float>(numberField(line, "rms").value_or(0.0));

        if (! record.path.empty())
            records.push_back(record);
    }

    return records;
}

bool keepRecordForSubset(const ManifestRecord& record, const std::string& subset)
{
    if (subset == "all")
        return true;
    if (subset != "pilot")
        throw std::runtime_error("Unknown subset: " + subset);

    static const std::unordered_set<int> pilotNotes { 21, 36, 60, 69, 84 };
    return pilotNotes.count(record.midiNote) > 0;
}

int findOnset(const std::vector<float>& samples)
{
    float peak = 0.0f;
    for (float sample : samples)
        peak = std::max(peak, std::abs(sample));

    const float threshold = std::max(peak * 0.01f, 1.0e-5f);
    for (size_t i = 0; i < samples.size(); ++i)
        if (std::abs(samples[i]) >= threshold)
            return static_cast<int>(i);
    return 0;
}

std::vector<float> trimToOnset(const std::vector<float>& samples, int sampleRate, int maxSamples)
{
    const int preRoll = sampleRate / 100;
    const int onset = findOnset(samples);
    const int start = std::max(0, onset - preRoll);
    const int available = static_cast<int>(samples.size()) - start;
    const int count = std::max(0, std::min(available, maxSamples));

    std::vector<float> out(static_cast<size_t>(count), 0.0f);
    if (count > 0)
        std::copy(samples.begin() + start, samples.begin() + start + count, out.begin());
    return out;
}

MonoSample loadMonoWav(const ManifestRecord& record, float maxSeconds)
{
    juce::AudioFormatManager manager;
    manager.registerBasicFormats();

    juce::File file(record.path);
    std::unique_ptr<juce::AudioFormatReader> reader(manager.createReaderFor(file));
    if (reader == nullptr)
        throw std::runtime_error("Could not read WAV: " + record.path);

    juce::AudioBuffer<float> buffer(static_cast<int>(reader->numChannels), static_cast<int>(reader->lengthInSamples));
    reader->read(&buffer, 0, static_cast<int>(reader->lengthInSamples), 0, true, true);

    std::vector<float> mono(static_cast<size_t>(buffer.getNumSamples()), 0.0f);
    for (int c = 0; c < buffer.getNumChannels(); ++c)
    {
        const float* input = buffer.getReadPointer(c);
        for (int i = 0; i < buffer.getNumSamples(); ++i)
            mono[static_cast<size_t>(i)] += input[i] / static_cast<float>(buffer.getNumChannels());
    }

    if (static_cast<int>(reader->sampleRate) != kSampleRate)
        throw std::runtime_error("Expected 48 kHz target: " + record.path);

    const int maxSamples = static_cast<int>(std::ceil(maxSeconds * kSampleRate));
    return { trimToOnset(mono, kSampleRate, maxSamples), kSampleRate };
}

float hzToMel(float hz)
{
    return 2595.0f * std::log10(1.0f + hz / 700.0f);
}

float melToHz(float mel)
{
    return 700.0f * (std::pow(10.0f, mel / 2595.0f) - 1.0f);
}

std::vector<std::vector<float>> makeMelFilters(int fftSize, int sampleRate, int bands)
{
    const int bins = fftSize / 2 + 1;
    std::vector<std::vector<float>> filters(static_cast<size_t>(bands), std::vector<float>(static_cast<size_t>(bins), 0.0f));
    const float minMel = hzToMel(20.0f);
    const float maxMel = hzToMel(static_cast<float>(sampleRate) * 0.5f);

    std::vector<int> points(static_cast<size_t>(bands + 2), 0);
    for (int i = 0; i < bands + 2; ++i)
    {
        const float mel = minMel + (maxMel - minMel) * static_cast<float>(i) / static_cast<float>(bands + 1);
        const float hz = melToHz(mel);
        points[static_cast<size_t>(i)] = std::clamp(static_cast<int>(std::floor((fftSize + 1) * hz / sampleRate)), 0, bins - 1);
    }

    for (int band = 0; band < bands; ++band)
    {
        const int left = points[static_cast<size_t>(band)];
        const int center = std::max(points[static_cast<size_t>(band + 1)], left + 1);
        const int right = std::max(points[static_cast<size_t>(band + 2)], center + 1);

        for (int bin = left; bin < center && bin < bins; ++bin)
            filters[static_cast<size_t>(band)][static_cast<size_t>(bin)] = static_cast<float>(bin - left) / static_cast<float>(center - left);
        for (int bin = center; bin < right && bin < bins; ++bin)
            filters[static_cast<size_t>(band)][static_cast<size_t>(bin)] = static_cast<float>(right - bin) / static_cast<float>(right - center);
    }

    return filters;
}

ScaleFeatures extractScaleFeatures(const std::vector<float>& samples, int sampleRate, int fftSize, int melBands)
{
    const int hop = fftSize / 4;
    const int bins = fftSize / 2 + 1;
    const int frameCount = std::max(1, 1 + static_cast<int>(std::ceil(std::max(0, static_cast<int>(samples.size()) - fftSize) / static_cast<float>(hop))));

    audiofft::AudioFFT fft;
    fft.init(static_cast<size_t>(fftSize));
    std::vector<float> frame(static_cast<size_t>(fftSize), 0.0f);
    std::vector<float> real(audiofft::AudioFFT::ComplexSize(static_cast<size_t>(fftSize)), 0.0f);
    std::vector<float> imag(real.size(), 0.0f);
    std::vector<float> magnitude(static_cast<size_t>(bins), 0.0f);
    std::vector<float> window(static_cast<size_t>(fftSize), 0.0f);
    auto filters = makeMelFilters(fftSize, sampleRate, melBands);

    for (int i = 0; i < fftSize; ++i)
        window[static_cast<size_t>(i)] = 0.5f - 0.5f * std::cos(static_cast<float>(2.0 * kPi * i / std::max(1, fftSize - 1)));

    ScaleFeatures features;
    features.fftSize = fftSize;
    features.melBands = melBands;
    features.melFrames.resize(static_cast<size_t>(frameCount * melBands), 0.0f);
    features.meanLogMel.resize(static_cast<size_t>(melBands), 0.0f);

    double centroidTotal = 0.0;
    double centroidWeight = 0.0;
    double energy = 0.0;

    for (int frameIndex = 0; frameIndex < frameCount; ++frameIndex)
    {
        const int start = frameIndex * hop;
        std::fill(frame.begin(), frame.end(), 0.0f);
        for (int i = 0; i < fftSize; ++i)
        {
            const int index = start + i;
            if (index < static_cast<int>(samples.size()))
                frame[static_cast<size_t>(i)] = samples[static_cast<size_t>(index)] * window[static_cast<size_t>(i)];
        }

        fft.fft(frame.data(), real.data(), imag.data());

        for (int bin = 0; bin < bins; ++bin)
        {
            float mag = std::sqrt(real[static_cast<size_t>(bin)] * real[static_cast<size_t>(bin)] + imag[static_cast<size_t>(bin)] * imag[static_cast<size_t>(bin)]);
            if (! std::isfinite(mag))
                mag = 0.0f;
            magnitude[static_cast<size_t>(bin)] = mag;
            const float hz = static_cast<float>(bin * sampleRate) / static_cast<float>(fftSize);
            centroidTotal += hz * mag;
            centroidWeight += mag;
        }

        for (int band = 0; band < melBands; ++band)
        {
            float mel = 0.0f;
            for (int bin = 0; bin < bins; ++bin)
                mel += filters[static_cast<size_t>(band)][static_cast<size_t>(bin)] * magnitude[static_cast<size_t>(bin)];
            if (! std::isfinite(mel) || mel < 0.0f)
                mel = 0.0f;
            features.melFrames[static_cast<size_t>(frameIndex * melBands + band)] = mel;
            features.meanLogMel[static_cast<size_t>(band)] += std::log(mel + kEpsilon);
            energy += mel * mel;
        }
    }

    for (float& value : features.meanLogMel)
        value /= static_cast<float>(frameCount);

    features.targetEnergy = static_cast<float>(std::sqrt(energy) + kEpsilon);
    features.centroid = static_cast<float>(centroidTotal / std::max(centroidWeight, 1.0e-9));
    return features;
}

std::vector<float> extractEnvelope(const std::vector<float>& samples)
{
    constexpr int window = 1024;
    constexpr int hop = 512;
    const int frames = std::max(1, 1 + static_cast<int>(std::ceil(std::max(0, static_cast<int>(samples.size()) - window) / static_cast<float>(hop))));
    std::vector<float> envelope(static_cast<size_t>(frames), 0.0f);

    for (int f = 0; f < frames; ++f)
    {
        const int start = f * hop;
        double sum = 0.0;
        int count = 0;
        for (int i = 0; i < window; ++i)
        {
            const int index = start + i;
            if (index < static_cast<int>(samples.size()))
            {
                const float value = samples[static_cast<size_t>(index)];
                sum += value * value;
                ++count;
            }
        }
        envelope[static_cast<size_t>(f)] = std::sqrt(static_cast<float>(sum / std::max(1, count)));
    }

    return envelope;
}

std::array<float, 13> computeMfcc(const std::vector<float>& meanLogMel)
{
    std::array<float, 13> mfcc {};
    const int bands = static_cast<int>(meanLogMel.size());
    for (int k = 0; k < static_cast<int>(mfcc.size()); ++k)
    {
        double sum = 0.0;
        for (int n = 0; n < bands; ++n)
            sum += meanLogMel[static_cast<size_t>(n)] * std::cos(kPi * k * (n + 0.5) / bands);
        mfcc[static_cast<size_t>(k)] = static_cast<float>(sum / std::sqrt(static_cast<double>(bands)));
    }
    return mfcc;
}

AudioFeatures extractFeatures(const std::vector<float>& samples, int sampleRate)
{
    AudioFeatures features;
    for (int fftSize : { 1024, 2048, 8192 })
        features.scales.push_back(extractScaleFeatures(samples, sampleRate, fftSize, 64));
    features.mfcc = computeMfcc(features.scales[1].meanLogMel);
    features.envelope = extractEnvelope(samples);
    return features;
}

float analyticGain(const std::vector<float>& target, const std::vector<float>& render)
{
    double dot = 0.0;
    double den = 0.0;
    const size_t count = std::min(target.size(), render.size());
    for (size_t i = 0; i < count; ++i)
    {
        dot += target[i] * render[i];
        den += render[i] * render[i];
    }
    const float gain = static_cast<float>(dot / std::max(den, 1.0e-12));
    return std::isfinite(gain) ? std::clamp(gain, 0.01f, 100.0f) : 1.0f;
}

float featureLoss(const AudioFeatures& target, const AudioFeatures& render)
{
    double loss = 0.0;

    for (size_t scaleIndex = 0; scaleIndex < target.scales.size(); ++scaleIndex)
    {
        const auto& t = target.scales[scaleIndex];
        const auto& r = render.scales[scaleIndex];
        const size_t count = std::min(t.melFrames.size(), r.melFrames.size());
        double logL1 = 0.0;
        double diffEnergy = 0.0;
        for (size_t i = 0; i < count; ++i)
        {
            const float tv = t.melFrames[i];
            const float rv = r.melFrames[i];
            const double logDiff = std::abs(std::log(std::max(rv, 0.0f) + kEpsilon) - std::log(std::max(tv, 0.0f) + kEpsilon));
            if (std::isfinite(logDiff))
                logL1 += logDiff;
            const float diff = rv - tv;
            if (std::isfinite(diff))
                diffEnergy += diff * diff;
        }

        loss += (count == 0 ? 0.0 : logL1 / static_cast<double>(count));
        loss += 0.5 * std::sqrt(diffEnergy) / std::max(static_cast<double>(t.targetEnergy), 1.0e-9);
    }

    loss += 0.1 * std::abs(target.scales[1].centroid - render.scales[1].centroid) / (kSampleRate * 0.5);

    double mfccLoss = 0.0;
    for (size_t i = 0; i < target.mfcc.size(); ++i)
    {
        const float diff = target.mfcc[i] - render.mfcc[i];
        if (std::isfinite(diff))
            mfccLoss += diff * diff;
    }
    loss += 0.05 * mfccLoss / static_cast<double>(target.mfcc.size());

    const size_t envCount = std::min(target.envelope.size(), render.envelope.size());
    double envLoss = 0.0;
    for (size_t i = 0; i < envCount; ++i)
    {
        const double diff = std::abs(std::log(std::max(target.envelope[i], 0.0f) + kEpsilon)
                                     - std::log(std::max(render.envelope[i], 0.0f) + kEpsilon));
        if (std::isfinite(diff))
            envLoss += diff;
    }
    loss += 0.15 * (envCount == 0 ? 0.0 : envLoss / static_cast<double>(envCount));

    return std::isfinite(loss) ? static_cast<float>(loss) : 1.0e9f;
}

struct FitModel
{
    std::vector<int> baseParams;
    std::vector<int> noteCurveParams {
        pStringDecay,
        pStringLopass,
        pHammerMass,
        pHammerSpringConstant,
        pHammerHysteresis,
        pBridgeImpedance,
        pLongitudinalMix,
        pStringDetuning,
    };
    std::vector<int> velocityCurveParams {
        pHammerMass,
        pHammerSpringConstant,
        pHammerHysteresis,
        pStringLopass,
        pLongitudinalMix,
        pMaxVelocity,
    };

    FitModel()
    {
        for (int i = 0; i < NumParams; ++i)
        {
            if (i == pVolume || i == pDwgs4 || i == pDownsample || i == pLongModes)
                continue;
            baseParams.push_back(i);
        }
    }

    int genomeSize() const
    {
        return static_cast<int>(baseParams.size() + noteCurveParams.size() * 2 + velocityCurveParams.size());
    }

    std::vector<float> initialGenome() const
    {
        return std::vector<float>(static_cast<size_t>(genomeSize()), 0.5f);
    }

    std::array<float, NumParams> parametersFor(const std::vector<float>& genome, int midiNote, float targetVelocity) const
    {
        std::array<float, NumParams> values {};
        values.fill(0.5f);
        values[pVolume] = 0.7f;
        values[pDwgs4] = 1.0f;
        values[pDownsample] = 0.0f;
        values[pLongModes] = 0.0f;

        size_t offset = 0;
        for (int param : baseParams)
            values[static_cast<size_t>(param)] = genome[offset++];

        const float noteX = std::clamp((static_cast<float>(midiNote) - 60.0f) / 48.0f, -1.2f, 1.2f);
        for (int param : noteCurveParams)
        {
            const float linear = (genome[offset++] - 0.5f) * 0.35f;
            const float quadratic = (genome[offset++] - 0.5f) * 0.25f;
            values[static_cast<size_t>(param)] += linear * noteX + quadratic * (noteX * noteX - 0.25f);
        }

        const float velocityX = std::clamp((targetVelocity - 0.65f) / 0.35f, -1.0f, 1.0f);
        for (int param : velocityCurveParams)
        {
            const float slope = (genome[offset++] - 0.5f) * 0.25f;
            values[static_cast<size_t>(param)] += slope * velocityX;
        }

        for (float& value : values)
            value = std::clamp(value, kFitParameterMin, kFitParameterMax);
        values[pDwgs4] = 1.0f;
        values[pDownsample] = 0.0f;
        values[pLongModes] = 0.0f;
        return values;
    }
};

int midiVelocityForLayer(float targetVelocity)
{
    return std::clamp(static_cast<int>(std::lround(std::sqrt(std::clamp(targetVelocity, 0.0f, 1.0f)) * 127.0f)), 1, 127);
}

float midiFrequency(int midiNote)
{
    return 440.0f * std::pow(2.0f, (static_cast<float>(midiNote) - 69.0f) / 12.0f);
}

float physicalParameterValue(int index, float value)
{
    switch (index)
    {
        case pYoungsModulus: return 200.0f * std::exp(4.0f * (value - 0.5f));
        case pStringDensity: return 7850.0f * std::exp(4.0f * (value - 0.5f));
        case pHammerMass: return std::exp(4.0f * (value - 0.5f));
        case pStringTension: return 800.0f * std::exp(3.0f * (value - 0.5f));
        case pStringLength: return std::exp(2.0f * (value - 0.25f));
        case pStringRadius: return std::exp(2.0f * (value - 0.25f));
        case pHammerCompliance: return 2.0f * value;
        case pHammerSpringConstant: return 2.0f * value;
        case pHammerHysteresis: return std::exp(4.0f * (value - 0.5f));
        case pBridgeImpedance: return 8000.0f * std::exp(12.0f * (value - 0.5f));
        case pBridgeHorizontalImpedance: return 60000.0f * std::exp(12.0f * (value - 0.5f));
        case pVerticalHorizontalImpedance: return 400.0f * std::exp(12.0f * (value - 0.5f));
        case pHammerPosition: return 0.05f + value * 0.15f;
        case pSoundboardSize: return value;
        case pStringDecay: return 0.25f * std::exp(6.0f * (value - 0.25f));
        case pStringLopass: return 5.85f * std::exp(6.0f * (value - 0.5f));
        case pDampedStringDecay: return 8.0f * std::exp(6.0f * (value - 0.5f));
        case pDampedStringLopass: return 25.0f * std::exp(6.0f * (value - 0.5f));
        case pSoundboardDecay: return 20.0f * std::exp(4.0f * (value - 0.5f));
        case pSoundboardLopass: return 20.0f * std::exp(4.0f * (value - 0.5f));
        case pLongitudinalGamma: return 1.0e-2f * std::exp(10.0f * (value - 0.5f));
        case pLongitudinalGammaQuadratic: return 1.0e-2f * std::exp(8.0f * (value - 0.5f));
        case pLongitudinalGammaDamped: return 5.0e-2f * std::exp(10.0f * (value - 0.5f));
        case pLongitudinalGammaQuadraticDamped: return 3.0e-2f * std::exp(8.0f * (value - 0.5f));
        case pLongitudinalMix: return value == 0.0f ? 0.0f : std::exp(16.0f * (value - 0.5f));
        case pLongitudinalTransverseMix: return value == 0.0f ? 0.0f : std::exp(16.0f * (value - 0.5f));
        case pVolume: return 5.0e-3f * std::exp(8.0f * (value - 0.5f));
        case pMaxVelocity: return 10.0f * std::exp(8.0f * (value - 0.5f));
        case pStringDetuning: return std::exp(10.0f * (value - 0.5f));
        case pBridgeMass: return 10.0f * std::exp(10.0f * (value - 0.5f));
        case pBridgeSpring: return 1.0e5f * std::exp(20.0f * (value - 0.5f));
        case pDwgs4: return std::lrint(value);
        case pDownsample: return 1.0f + std::lrint(value);
        case pLongModes: return 1.0f + std::lrint(value);
        default: return value;
    }
}

bool isGenomeStableForTarget(const FitModel& model, const std::vector<float>& genome, const ManifestRecord& record)
{
    std::array<float, NumParams> normalized = model.parametersFor(genome, record.midiNote, record.targetVelocity);
    std::array<float, NumParams> v {};
    for (int i = 0; i < NumParams; ++i)
        v[static_cast<size_t>(i)] = physicalParameterValue(i, normalized[static_cast<size_t>(i)]);

    const float f0 = 27.5f;
    const float f = midiFrequency(record.midiNote);
    const float logFrequency = std::log(f / f0);
    float length = 0.04f + 2.0f / (1.0f + std::exp(-3.2f + 1.4f * logFrequency));
    length *= v[static_cast<size_t>(pStringLength)];

    const float radiusBase = 0.008f * std::pow(3.0f + 1.5f * logFrequency, -1.4f);
    const float radius = radiusBase * v[static_cast<size_t>(pStringRadius)];
    const float area = static_cast<float>(kPi) * radius * radius;
    const float density = v[static_cast<size_t>(pStringDensity)];
    const float mu = area * density;
    const float tension = (2.0f * length * f) * (2.0f * length * f) * mu;
    const float coreRadius = radius < 0.0006f ? radius : 0.0006f;
    const float youngsModulus = v[static_cast<size_t>(pYoungsModulus)] * 1.0e9f;
    const float bending = static_cast<float>(kPi * kPi * kPi) * youngsModulus * std::pow(coreRadius, 4.0f)
        / (4.0f * length * length * tension);
    const float longitudinalSpeed = std::sqrt(youngsModulus / density);
    const float longitudinalFundamental = longitudinalSpeed / (2.0f * length);
    const int forcedLongModes = 2;
    const int nLongModes = static_cast<int>(0.5f * kSampleRate / forcedLongModes / longitudinalFundamental - 0.5f);
    if (nLongModes >= nMaxLongModes)
        return false;

    const int downsample = 1;
    int nstrings = 3;
    if (record.midiNote < 31)
        nstrings = 1;
    else if (record.midiNote < 41)
        nstrings = 2;

    static constexpr float tune[3][3] {
        { 1.0f, 0.0f, 0.0f },
        { 0.9997f, 1.0003f, 0.0f },
        { 1.0001f, 1.0003f, 0.9996f },
    };

    float hammerPosition = v[static_cast<size_t>(pHammerPosition)];
    hammerPosition = hammerPosition / (1.0f + 0.01f * std::pow(logFrequency, 2.0f));

    for (int stringIndex = 0; stringIndex < nstrings; ++stringIndex)
    {
        const float fk = f * (1.0f + (tune[nstrings - 1][stringIndex] - 1.0f) * v[static_cast<size_t>(pStringDetuning)]);
        dwgs string;
        const int upsample = string.getMinUpsample(downsample, static_cast<float>(kSampleRate), fk, hammerPosition, bending);
        const float deltot = static_cast<float>(kSampleRate) / static_cast<float>(downsample) / fk * static_cast<float>(upsample);
        const int del0 = static_cast<int>(0.5f * (hammerPosition * deltot));
        int del2 = static_cast<int>(0.5f * (deltot - hammerPosition * deltot) - 1.0f);
        if (del2 < 1)
            return false;

        const float delHalf = 0.5f * deltot;
        const float delHammerHalf = 0.5f * hammerPosition * deltot;
        float dTop = delHalf - delHammerHalf - static_cast<float>(del2);
        const int dd = std::min(4, del2 - 1);
        dTop += static_cast<float>(dd);
        del2 -= dd;
        const int del4 = static_cast<int>(dTop);
        const int delTab = del0 + del2 + del4;
        if (del0 < 0 || del2 < 0 || del4 < 0 || delTab < 0)
            return false;
        if (del0 >= DelaySize || del2 >= DelaySize || del4 >= DelaySize || delTab >= DelaySize)
            return false;
    }

    return true;
}

bool isGenomeStable(const FitModel& model, const std::vector<float>& genome, const std::vector<TargetExample>& targets)
{
    return std::all_of(targets.begin(), targets.end(), [&](const TargetExample& target) {
        return isGenomeStableForTarget(model, genome, target.record);
    });
}

std::vector<float> renderPiano(const FitModel& model, const std::vector<float>& genome, const ManifestRecord& record, int sampleCount)
{
    Piano piano;
    piano.init(static_cast<float>(kSampleRate), kBlockSize);

    const auto params = model.parametersFor(genome, record.midiNote, record.targetVelocity);
    for (int i = 0; i < NumParams; ++i)
        piano.setParameter(i, params[static_cast<size_t>(i)]);
    for (int param : { pSoundboardSize, pSoundboardDecay, pSoundboardLopass })
        piano.setParameter(param, params[static_cast<size_t>(param)]);

    const int renderSamples = sampleCount + kSampleRate;
    juce::AudioBuffer<float> output(2, renderSamples);
    output.clear();

    int currentSample = 0;
    bool noteOnSent = false;
    while (currentSample < renderSamples)
    {
        const int samplesToProcess = std::min(kBlockSize, renderSamples - currentSample);
        juce::MidiBuffer midi;
        if (! noteOnSent)
        {
            midi.addEvent(juce::MidiMessage::noteOn(1, record.midiNote, static_cast<juce::uint8>(midiVelocityForLayer(record.targetVelocity))), 0);
            noteOnSent = true;
        }

        float* channels[2] {
            output.getWritePointer(0, currentSample),
            output.getWritePointer(1, currentSample),
        };
        piano.process(channels, samplesToProcess, midi);
        currentSample += samplesToProcess;
    }

    std::vector<float> mono(static_cast<size_t>(renderSamples), 0.0f);
    const float* left = output.getReadPointer(0);
    const float* right = output.getReadPointer(1);
    for (int i = 0; i < renderSamples; ++i)
    {
        float sample = 0.5f * (left[i] + right[i]);
        mono[static_cast<size_t>(i)] = std::isfinite(sample) ? sample : 0.0f;
    }

    auto aligned = trimToOnset(mono, kSampleRate, sampleCount);
    aligned.resize(static_cast<size_t>(sampleCount), 0.0f);
    return aligned;
}

float evaluateGenome(const FitModel& model, const std::vector<float>& genome, const std::vector<TargetExample>& targets)
{
    if (! isGenomeStable(model, genome, targets))
        return kInvalidGenomeLoss;

    double total = 0.0;
    for (const auto& target : targets)
    {
        auto render = renderPiano(model, genome, target.record, static_cast<int>(target.audio.samples.size()));
        const float gain = analyticGain(target.audio.samples, render);
        for (float& sample : render)
            sample *= gain;
        const auto renderFeatures = extractFeatures(render, kSampleRate);
        total += featureLoss(target.features, renderFeatures);
    }
    const float loss = static_cast<float>(total / std::max<size_t>(1, targets.size()));
    return std::isfinite(loss) ? loss : kInvalidGenomeLoss;
}

struct Candidate
{
    std::vector<float> genome;
    float loss = std::numeric_limits<float>::infinity();
};

struct ExternalGenome
{
    std::string id;
    std::vector<float> genome;
};

struct JsonValue
{
    enum class Type
    {
        Null,
        Bool,
        Number,
        String,
        Array,
        Object
    };

    Type type = Type::Null;
    bool boolValue = false;
    double numberValue = 0.0;
    std::string stringValue;
    std::vector<JsonValue> arrayValue;
    std::vector<std::pair<std::string, JsonValue>> objectValue;

    const JsonValue* field(const std::string& key) const
    {
        if (type != Type::Object)
            return nullptr;
        for (const auto& item : objectValue)
            if (item.first == key)
                return &item.second;
        return nullptr;
    }
};

class JsonParser
{
public:
    explicit JsonParser(std::string textToParse)
        : text(std::move(textToParse))
    {
    }

    JsonValue parse()
    {
        auto value = parseValue();
        skipWhitespace();
        if (position != text.size())
            throw std::runtime_error("Unexpected trailing JSON input");
        return value;
    }

private:
    void skipWhitespace()
    {
        while (position < text.size() && std::isspace(static_cast<unsigned char>(text[position])) != 0)
            ++position;
    }

    char peek()
    {
        skipWhitespace();
        if (position >= text.size())
            throw std::runtime_error("Unexpected end of JSON input");
        return text[position];
    }

    bool consume(char expected)
    {
        skipWhitespace();
        if (position < text.size() && text[position] == expected)
        {
            ++position;
            return true;
        }
        return false;
    }

    void expect(char expected)
    {
        if (! consume(expected))
            throw std::runtime_error(std::string("Expected JSON character: ") + expected);
    }

    JsonValue parseValue()
    {
        const char c = peek();
        if (c == '"')
            return JsonValue { JsonValue::Type::String, false, 0.0, parseString(), {}, {} };
        if (c == '[')
            return parseArray();
        if (c == '{')
            return parseObject();
        if (c == '-' || std::isdigit(static_cast<unsigned char>(c)) != 0)
            return parseNumber();
        if (text.compare(position, 4, "true") == 0)
        {
            position += 4;
            JsonValue value;
            value.type = JsonValue::Type::Bool;
            value.boolValue = true;
            return value;
        }
        if (text.compare(position, 5, "false") == 0)
        {
            position += 5;
            JsonValue value;
            value.type = JsonValue::Type::Bool;
            return value;
        }
        if (text.compare(position, 4, "null") == 0)
        {
            position += 4;
            return {};
        }
        throw std::runtime_error("Expected JSON value");
    }

    std::string parseString()
    {
        expect('"');
        std::string out;
        while (position < text.size())
        {
            const char c = text[position++];
            if (c == '"')
                return out;
            if (c != '\\')
            {
                out.push_back(c);
                continue;
            }
            if (position >= text.size())
                throw std::runtime_error("Unterminated JSON escape");
            const char escaped = text[position++];
            switch (escaped)
            {
                case '"': out.push_back('"'); break;
                case '\\': out.push_back('\\'); break;
                case '/': out.push_back('/'); break;
                case 'b': out.push_back('\b'); break;
                case 'f': out.push_back('\f'); break;
                case 'n': out.push_back('\n'); break;
                case 'r': out.push_back('\r'); break;
                case 't': out.push_back('\t'); break;
                default: throw std::runtime_error("Unsupported JSON escape");
            }
        }
        throw std::runtime_error("Unterminated JSON string");
    }

    JsonValue parseNumber()
    {
        const size_t start = position;
        if (text[position] == '-')
            ++position;
        while (position < text.size() && std::isdigit(static_cast<unsigned char>(text[position])) != 0)
            ++position;
        if (position < text.size() && text[position] == '.')
        {
            ++position;
            while (position < text.size() && std::isdigit(static_cast<unsigned char>(text[position])) != 0)
                ++position;
        }
        if (position < text.size() && (text[position] == 'e' || text[position] == 'E'))
        {
            ++position;
            if (position < text.size() && (text[position] == '+' || text[position] == '-'))
                ++position;
            while (position < text.size() && std::isdigit(static_cast<unsigned char>(text[position])) != 0)
                ++position;
        }

        JsonValue value;
        value.type = JsonValue::Type::Number;
        value.numberValue = std::stod(text.substr(start, position - start));
        return value;
    }

    JsonValue parseArray()
    {
        expect('[');
        JsonValue value;
        value.type = JsonValue::Type::Array;
        if (consume(']'))
            return value;
        do
        {
            value.arrayValue.push_back(parseValue());
        } while (consume(','));
        expect(']');
        return value;
    }

    JsonValue parseObject()
    {
        expect('{');
        JsonValue value;
        value.type = JsonValue::Type::Object;
        if (consume('}'))
            return value;
        do
        {
            std::string key = parseString();
            expect(':');
            value.objectValue.push_back({ std::move(key), parseValue() });
        } while (consume(','));
        expect('}');
        return value;
    }

    std::string text;
    size_t position = 0;
};

std::string requireStringField(const JsonValue& value, const std::string& key)
{
    const auto* field = value.field(key);
    if (field == nullptr || field->type != JsonValue::Type::String)
        throw std::runtime_error("Expected string field: " + key);
    return field->stringValue;
}

std::optional<std::string> optionalStringField(const JsonValue& value, const std::string& key)
{
    const auto* field = value.field(key);
    if (field == nullptr)
        return std::nullopt;
    if (field->type != JsonValue::Type::String)
        throw std::runtime_error("Expected string field: " + key);
    return field->stringValue;
}

std::vector<ExternalGenome> parseEvaluateGenomes(const JsonValue& request, int expectedSize)
{
    const auto* genomesValue = request.field("genomes");
    if (genomesValue == nullptr || genomesValue->type != JsonValue::Type::Array)
        throw std::runtime_error("Expected genomes array");

    std::vector<ExternalGenome> genomes;
    genomes.reserve(genomesValue->arrayValue.size());
    for (const auto& itemValue : genomesValue->arrayValue)
    {
        if (itemValue.type != JsonValue::Type::Object)
            throw std::runtime_error("Expected genome object");

        ExternalGenome item;
        item.id = requireStringField(itemValue, "id");
        const auto* genomeValue = itemValue.field("genome");
        if (genomeValue == nullptr || genomeValue->type != JsonValue::Type::Array)
            throw std::runtime_error("Expected genome array");

        item.genome.reserve(genomeValue->arrayValue.size());
        for (const auto& numberValue : genomeValue->arrayValue)
        {
            if (numberValue.type != JsonValue::Type::Number)
                throw std::runtime_error("Genome values must be numbers");
            item.genome.push_back(std::clamp(static_cast<float>(numberValue.numberValue), 0.0f, 1.0f));
        }

        if (static_cast<int>(item.genome.size()) != expectedSize)
        {
            std::ostringstream message;
            message << "Genome id " << item.id << " has " << item.genome.size()
                    << " values, expected " << expectedSize;
            throw std::runtime_error(message.str());
        }
        genomes.push_back(std::move(item));
    }

    if (genomes.empty())
        throw std::runtime_error("Evaluate request contains no genomes");
    return genomes;
}

std::vector<float> parseNumberArray(const std::string& text, size_t start)
{
    const auto open = text.find('[', start);
    if (open == std::string::npos)
        throw std::runtime_error("Expected genome array");
    const auto close = text.find(']', open);
    if (close == std::string::npos)
        throw std::runtime_error("Unterminated genome array");

    const std::string arrayText = text.substr(open + 1, close - open - 1);
    const std::regex numberRe("-?[0-9]+(?:\\.[0-9]+)?(?:[eE][+-]?[0-9]+)?");
    std::vector<float> values;
    for (std::sregex_iterator it(arrayText.begin(), arrayText.end(), numberRe), end; it != end; ++it)
        values.push_back(std::stof((*it)[0].str()));
    return values;
}

std::vector<ExternalGenome> readExternalGenomes(const std::string& path, int expectedSize)
{
    std::ifstream in(path);
    if (! in)
        throw std::runtime_error("Could not open genome input: " + path);

    std::vector<ExternalGenome> genomes;
    std::string line;
    int lineNumber = 0;
    while (std::getline(in, line))
    {
        ++lineNumber;
        if (line.empty())
            continue;

        ExternalGenome item;
        item.id = stringField(line, "id").value_or(std::to_string(lineNumber - 1));
        const auto genomeKey = line.find("\"genome\"");
        item.genome = parseNumberArray(line, genomeKey == std::string::npos ? 0 : genomeKey);
        if (static_cast<int>(item.genome.size()) != expectedSize)
        {
            std::ostringstream message;
            message << "Genome on line " << lineNumber << " has " << item.genome.size()
                    << " values, expected " << expectedSize;
            throw std::runtime_error(message.str());
        }
        for (float& value : item.genome)
            value = std::clamp(value, 0.0f, 1.0f);
        genomes.push_back(std::move(item));
    }

    if (genomes.empty())
        throw std::runtime_error("No genomes found in: " + path);
    return genomes;
}

std::vector<float> readOneExternalGenome(const std::string& path, int expectedSize)
{
    auto genomes = readExternalGenomes(path, expectedSize);
    return genomes.front().genome;
}

class MetricsLogger
{
public:
    explicit MetricsLogger(const std::string& path)
    {
        if (! path.empty())
        {
            out.open(path);
            if (! out)
                throw std::runtime_error("Could not write metrics: " + path);
            enabled = true;
        }
    }

    void logConfig(const Options& options, int targetCount, int genomeSize)
    {
        std::lock_guard<std::mutex> lock(mutex);
        if (! enabled)
            return;

        out << "{\"type\":\"config\""
            << ",\"optimizer\":\"pagmo.pso_gen\""
            << ",\"subset\":\"" << jsonEscape(options.subset) << "\""
            << ",\"target_count\":" << targetCount
            << ",\"genome_size\":" << genomeSize
            << ",\"population\":" << options.population
            << ",\"omega\":" << options.omega
            << ",\"eta1\":" << options.eta1
            << ",\"eta2\":" << options.eta2
            << ",\"max_velocity\":" << options.maxVelocity
            << ",\"pso_variant\":" << options.psoVariant
            << ",\"neighbourhood_type\":" << options.neighbourhoodType
            << ",\"neighbourhood_param\":" << options.neighbourhoodParam
            << ",\"pso_verbosity\":" << options.psoVerbosity
            << ",\"pso_memory\":" << (options.psoMemory ? "true" : "false")
            << ",\"parallel_evaluations\":" << (options.parallelEvaluations ? "true" : "false")
            << ",\"fit_parameter_min\":" << kFitParameterMin
            << ",\"fit_parameter_max\":" << kFitParameterMax
            << ",\"max_evaluations\":" << options.maxEvaluations
            << ",\"max_seconds\":" << options.maxSeconds
            << ",\"seed\":" << options.seed
            << "}\n";
        out.flush();
    }

    void logEvaluation(int evaluation, float loss, float bestLoss, const char* phase)
    {
        std::lock_guard<std::mutex> lock(mutex);
        if (! enabled)
            return;

        out << std::setprecision(9)
            << "{\"type\":\"evaluation\""
            << ",\"evaluation\":" << evaluation
            << ",\"loss\":" << loss
            << ",\"best_loss\":" << bestLoss
            << ",\"phase\":\"" << phase << "\""
            << "}\n";
        out.flush();
    }

    void logGeneration(int evaluations,
                       int generation,
                       float bestLoss,
                       double meanVelocity,
                       double meanLocalBest,
                       double averageDistance)
    {
        std::lock_guard<std::mutex> lock(mutex);
        if (! enabled)
            return;

        out << std::setprecision(9)
            << "{\"type\":\"generation\""
            << ",\"evaluation\":" << evaluations
            << ",\"generation\":" << generation
            << ",\"best_loss\":" << bestLoss
            << ",\"mean_velocity\":" << meanVelocity
            << ",\"mean_lbest\":" << meanLocalBest
            << ",\"avg_distance\":" << averageDistance
            << "}\n";
        out.flush();
    }

    void logFinal(float initialLoss, float bestLoss)
    {
        std::lock_guard<std::mutex> lock(mutex);
        if (! enabled)
            return;

        out << std::setprecision(9)
            << "{\"type\":\"final\""
            << ",\"initial_loss\":" << initialLoss
            << ",\"best_loss\":" << bestLoss
            << ",\"improvement\":" << (initialLoss - bestLoss)
            << "}\n";
        out.flush();
    }

private:
    bool enabled = false;
    std::ofstream out;
    std::mutex mutex;
};

void evaluateExternalGenomes(const Options& options,
                             const FitModel& model,
                             const std::vector<TargetExample>& targets,
                             MetricsLogger& metrics)
{
    const auto genomes = readExternalGenomes(options.evalGenomesPath, model.genomeSize());
    std::ofstream out(options.evalOutputPath);
    if (! out)
        throw std::runtime_error("Could not write genome losses: " + options.evalOutputPath);

    float bestLoss = std::numeric_limits<float>::infinity();
    int evaluation = 0;
    for (const auto& item : genomes)
    {
        const float loss = evaluateGenome(model, item.genome, targets);
        ++evaluation;
        bestLoss = std::min(bestLoss, loss);
        metrics.logEvaluation(evaluation, loss, bestLoss, "external");
        out << std::setprecision(9)
            << "{\"id\":\"" << jsonEscape(item.id) << "\",\"loss\":" << loss << "}\n";
    }
}

void writeErrorResponse(const std::optional<std::string>& batchId, const std::string& message)
{
    std::cout << "{\"type\":\"error\"";
    if (batchId.has_value())
        std::cout << ",\"batch_id\":\"" << jsonEscape(*batchId) << "\"";
    std::cout << ",\"message\":\"" << jsonEscape(message) << "\"}\n";
    std::cout.flush();
}

void serveJsonlEvaluator(const FitModel& model,
                         const std::vector<TargetExample>& targets,
                         MetricsLogger& metrics)
{
    std::string line;
    int evaluation = 0;
    float bestLoss = std::numeric_limits<float>::infinity();

    while (std::getline(std::cin, line))
    {
        if (line.empty())
            continue;

        std::optional<std::string> batchId;
        try
        {
            const auto request = JsonParser(line).parse();
            if (request.type != JsonValue::Type::Object)
                throw std::runtime_error("Request must be a JSON object");

            batchId = optionalStringField(request, "batch_id");
            const auto type = requireStringField(request, "type");
            if (type == "shutdown")
                return;
            if (type != "evaluate")
                throw std::runtime_error("Unknown request type: " + type);

            if (! batchId.has_value())
                throw std::runtime_error("Evaluate request requires batch_id");
            const auto genomes = parseEvaluateGenomes(request, model.genomeSize());

            std::cout << std::setprecision(9)
                      << "{\"type\":\"result\",\"batch_id\":\"" << jsonEscape(*batchId) << "\",\"losses\":[";
            for (size_t i = 0; i < genomes.size(); ++i)
            {
                const auto& item = genomes[i];
                const float loss = evaluateGenome(model, item.genome, targets);
                ++evaluation;
                bestLoss = std::min(bestLoss, loss);
                metrics.logEvaluation(evaluation, loss, bestLoss, "external-stdio");
                if (i != 0)
                    std::cout << ",";
                std::cout << "{\"id\":\"" << jsonEscape(item.id) << "\",\"loss\":" << loss << "}";
            }
            std::cout << "]}\n";
            std::cout.flush();
        }
        catch (const std::exception& e)
        {
            writeErrorResponse(batchId, e.what());
        }
    }
}

std::vector<float> toGenome(const pagmo::vector_double& values)
{
    std::vector<float> genome;
    genome.reserve(values.size());
    for (double value : values)
        genome.push_back(static_cast<float>(std::clamp(value, 0.0, 1.0)));
    return genome;
}

pagmo::vector_double toDecisionVector(const std::vector<float>& genome)
{
    pagmo::vector_double values;
    values.reserve(genome.size());
    for (float value : genome)
        values.push_back(static_cast<double>(std::clamp(value, 0.0f, 1.0f)));
    return values;
}

pagmo::vector_double flattenDecisionVectors(const std::vector<pagmo::vector_double>& decisionVectors)
{
    pagmo::vector_double flattened;
    size_t totalSize = 0;
    for (const auto& decisionVector : decisionVectors)
        totalSize += decisionVector.size();
    flattened.reserve(totalSize);

    for (const auto& decisionVector : decisionVectors)
        flattened.insert(flattened.end(), decisionVector.begin(), decisionVector.end());
    return flattened;
}

std::vector<double> evaluateDecisionVectors(const pagmo::problem& problem,
                                            const std::vector<pagmo::vector_double>& decisionVectors,
                                            bool parallelEvaluations)
{
    std::vector<double> losses;
    losses.reserve(decisionVectors.size());

    if (decisionVectors.empty())
        return losses;

    if (parallelEvaluations && decisionVectors.size() > 1)
    {
        pagmo::bfe evaluator { pagmo::thread_bfe {} };
        const auto fitnesses = evaluator(problem, flattenDecisionVectors(decisionVectors));
        if (fitnesses.size() != decisionVectors.size())
            throw std::runtime_error("pagmo thread_bfe returned an unexpected fitness vector size");
        losses.insert(losses.end(), fitnesses.begin(), fitnesses.end());
        return losses;
    }

    for (const auto& decisionVector : decisionVectors)
    {
        const auto fitness = problem.fitness(decisionVector);
        if (fitness.empty())
            throw std::runtime_error("pagmo fitness returned an empty vector");
        losses.push_back(fitness.front());
    }
    return losses;
}

struct PsoEvaluationState
{
    MetricsLogger* metrics = nullptr;
    mutable std::mutex mutex;
    int evaluations = 0;
    float bestLoss = std::numeric_limits<float>::infinity();

    void record(float loss, const char* phase)
    {
        int evaluation = 0;
        float currentBest = 0.0f;
        {
            std::lock_guard<std::mutex> lock(mutex);
            evaluation = ++evaluations;
            bestLoss = std::min(bestLoss, loss);
            currentBest = bestLoss;
        }

        if (metrics != nullptr)
            metrics->logEvaluation(evaluation, loss, currentBest, phase);
    }

    std::pair<int, float> snapshot() const
    {
        std::lock_guard<std::mutex> lock(mutex);
        return { evaluations, bestLoss };
    }
};

struct PianoFitPagmoProblem
{
    const FitModel* model = nullptr;
    const std::vector<TargetExample>* targets = nullptr;
    int genomeDimensions = 0;
    std::shared_ptr<PsoEvaluationState> state;

    pagmo::vector_double fitness(const pagmo::vector_double& values) const
    {
        if (model == nullptr || targets == nullptr || state == nullptr)
            throw std::runtime_error("PianoFit pagmo problem is not initialized");

        const auto genome = toGenome(values);
        const float loss = evaluateGenome(*model, genome, *targets);
        state->record(loss, "pagmo-pso-batch");
        return { static_cast<double>(loss) };
    }

    std::pair<pagmo::vector_double, pagmo::vector_double> get_bounds() const
    {
        return {
            pagmo::vector_double(static_cast<size_t>(genomeDimensions), static_cast<double>(kFitParameterMin)),
            pagmo::vector_double(static_cast<size_t>(genomeDimensions), static_cast<double>(kFitParameterMax)),
        };
    }

    std::string get_name() const
    {
        return "PianoFit pagmo generational PSO";
    }

    pagmo::thread_safety get_thread_safety() const
    {
        return pagmo::thread_safety::basic;
    }
};

Candidate runPagmoPso(const Options& options,
                      const FitModel& model,
                      const std::vector<TargetExample>& targets,
                      const std::vector<float>& initial,
                      float initialLoss,
                      MetricsLogger& metrics,
                      int& evaluations,
                      float& bestLoss)
{
    auto state = std::make_shared<PsoEvaluationState>();
    state->metrics = &metrics;
    state->evaluations = evaluations;
    state->bestLoss = bestLoss;

    PianoFitPagmoProblem udp {
        &model,
        &targets,
        model.genomeSize(),
        state,
    };

    pagmo::problem problem { udp };
    pagmo::population population { problem, 0u, options.seed };
    population.push_back(toDecisionVector(initial), { static_cast<double>(initialLoss) });

    std::vector<pagmo::vector_double> randomDecisionVectors;
    while (population.size() + randomDecisionVectors.size() < static_cast<pagmo::population::size_type>(options.population))
        randomDecisionVectors.push_back(population.random_decision_vector());

    const auto randomLosses = evaluateDecisionVectors(problem, randomDecisionVectors, options.parallelEvaluations);
    for (size_t i = 0; i < randomDecisionVectors.size(); ++i)
        population.push_back(randomDecisionVectors[i], { randomLosses[i] });

    const int evaluationsAfterPopulation = state->snapshot().first;
    const int remainingEvaluations = std::max(0, options.maxEvaluations - evaluationsAfterPopulation);
    const unsigned generations = static_cast<unsigned>(
        std::ceil(static_cast<double>(remainingEvaluations) / static_cast<double>(options.population)));

    if (generations > 0)
    {
        pagmo::pso_gen pso(
            generations,
            options.omega,
            options.eta1,
            options.eta2,
            options.maxVelocity,
            options.psoVariant,
            options.neighbourhoodType,
            options.neighbourhoodParam,
            options.psoMemory,
            options.seed);
        pso.set_verbosity(options.psoVerbosity);
        if (options.parallelEvaluations)
            pso.set_bfe(pagmo::bfe { pagmo::thread_bfe {} });

        pagmo::algorithm algorithm { pso };
        const int evaluationsBeforeEvolve = state->snapshot().first;
        population = algorithm.evolve(population);

        if (auto* evolvedPso = algorithm.extract<pagmo::pso_gen>())
        {
            for (const auto& line : evolvedPso->get_log())
            {
                const auto generation = static_cast<int>(std::get<0>(line));
                const auto pagmoEvaluations = static_cast<int>(std::get<1>(line));
                const auto generationEvaluation = evaluationsBeforeEvolve + pagmoEvaluations;
                metrics.logGeneration(
                    generationEvaluation,
                    generation,
                    static_cast<float>(std::get<2>(line)),
                    std::get<3>(line),
                    std::get<4>(line),
                    std::get<5>(line));
            }
        }
    }

    const auto [finalEvaluations, finalBestLoss] = state->snapshot();
    evaluations = finalEvaluations;
    bestLoss = finalBestLoss;

    auto championGenome = toGenome(population.champion_x());
    const auto championFitness = population.champion_f();
    const float championLoss = championFitness.empty()
        ? evaluateGenome(model, championGenome, targets)
        : static_cast<float>(championFitness.front());
    return { std::move(championGenome), championLoss };
}

bool writeMonoWav(const std::string& path, const std::vector<float>& samples)
{
    juce::File file(path);
    file.getParentDirectory().createDirectory();
    juce::WavAudioFormat format;
    std::unique_ptr<juce::FileOutputStream> stream(file.createOutputStream());
    if (stream == nullptr)
        return false;

    std::unique_ptr<juce::AudioFormatWriter> writer(format.createWriterFor(stream.get(), kSampleRate, 1, 32, {}, 0));
    if (writer == nullptr)
        return false;
    stream.release();

    juce::AudioBuffer<float> buffer(1, static_cast<int>(samples.size()));
    std::copy(samples.begin(), samples.end(), buffer.getWritePointer(0));
    return writer->writeFromAudioSampleBuffer(buffer, 0, buffer.getNumSamples());
}

void exportExamples(const std::string& dir, const FitModel& model, const std::vector<float>& genome, const std::vector<TargetExample>& targets)
{
    if (dir.empty())
        return;

    int exported = 0;
    for (const auto& target : targets)
    {
        if (exported >= 12)
            break;

        auto render = renderPiano(model, genome, target.record, static_cast<int>(target.audio.samples.size()));
        const float gain = analyticGain(target.audio.samples, render);
        for (float& sample : render)
            sample *= gain;

        const std::string stem = "midi" + std::to_string(target.record.midiNote) + "_" + std::string(1, target.record.layer);
        writeMonoWav(dir + "/" + stem + "_target.wav", target.audio.samples);
        writeMonoWav(dir + "/" + stem + "_render.wav", render);
        ++exported;
    }
}

void writeResultJson(const std::string& path,
                     const FitModel& model,
                     const std::vector<float>& genome,
                     float initialLoss,
                     float bestLoss,
                     const Options& options)
{
    std::ofstream out(path);
    if (! out)
        throw std::runtime_error("Could not write result: " + path);

    out << "{\n";
    out << "  \"initial_loss\": " << initialLoss << ",\n";
    out << "  \"best_loss\": " << bestLoss << ",\n";
    out << "  \"optimizer\": \"pagmo.pso_gen\",\n";
    out << "  \"subset\": \"" << jsonEscape(options.subset) << "\",\n";
    out << "  \"population\": " << options.population << ",\n";
    out << "  \"omega\": " << options.omega << ",\n";
    out << "  \"eta1\": " << options.eta1 << ",\n";
    out << "  \"eta2\": " << options.eta2 << ",\n";
    out << "  \"max_velocity\": " << options.maxVelocity << ",\n";
    out << "  \"pso_variant\": " << options.psoVariant << ",\n";
    out << "  \"neighbourhood_type\": " << options.neighbourhoodType << ",\n";
    out << "  \"neighbourhood_param\": " << options.neighbourhoodParam << ",\n";
    out << "  \"pso_verbosity\": " << options.psoVerbosity << ",\n";
    out << "  \"pso_memory\": " << (options.psoMemory ? "true" : "false") << ",\n";
    out << "  \"parallel_evaluations\": " << (options.parallelEvaluations ? "true" : "false") << ",\n";
    out << "  \"fit_parameter_min\": " << kFitParameterMin << ",\n";
    out << "  \"fit_parameter_max\": " << kFitParameterMax << ",\n";
    out << "  \"max_evaluations\": " << options.maxEvaluations << ",\n";
    out << "  \"genome\": [";
    for (size_t i = 0; i < genome.size(); ++i)
    {
        if (i != 0)
            out << ", ";
        out << genome[i];
    }
    out << "],\n";
    out << "  \"base_params\": {\n";
    const auto paramValues = model.parametersFor(genome, 60, 0.65f);
    for (int i = 0; i < NumParams; ++i)
    {
        out << "    \"" << ::params[i].name << "\": " << paramValues[static_cast<size_t>(i)];
        out << (i + 1 == NumParams ? "\n" : ",\n");
    }
    out << "  }\n";
    out << "}\n";
}

std::vector<TargetExample> loadTargets(const Options& options)
{
    const auto records = readManifest(options.manifestPath);
    std::vector<TargetExample> targets;

    for (const auto& record : records)
    {
        if (! keepRecordForSubset(record, options.subset))
            continue;

        TargetExample target;
        target.record = record;
        target.audio = loadMonoWav(record, options.maxSeconds);
        target.features = extractFeatures(target.audio.samples, target.audio.sampleRate);
        if (! target.audio.samples.empty())
            targets.push_back(std::move(target));
    }

    if (targets.empty())
        throw std::runtime_error("No targets selected from manifest");

    return targets;
}

} // namespace

int main(int argc, char** argv)
{
    try
    {
        const auto options = parseOptions(argc, argv);
        FitModel model;
        if (options.printModelInfo)
        {
            std::cout << "{\"genome_size\":" << model.genomeSize() << ",\"initial_value\":0.5}\n";
            return 0;
        }

        const auto targets = loadTargets(options);
        auto initial = model.initialGenome();
        MetricsLogger metrics(options.metricsPath);
        metrics.logConfig(options, static_cast<int>(targets.size()), model.genomeSize());

        auto& infoOut = options.serveJsonl ? std::cerr : std::cout;
        infoOut << "Loaded " << targets.size() << " target samples\n";
        infoOut << "Genome dimensions: " << model.genomeSize() << "\n";

        if (options.serveJsonl)
        {
            serveJsonlEvaluator(model, targets, metrics);
            return 0;
        }

        if (! options.evalGenomesPath.empty())
        {
            evaluateExternalGenomes(options, model, targets, metrics);
            return 0;
        }

        const float initialLoss = evaluateGenome(model, initial, targets);
        int evaluation = 1;
        float bestLoss = initialLoss;
        metrics.logEvaluation(evaluation, initialLoss, bestLoss, "initial");
        std::cout << "Initial loss: " << initialLoss << "\n";

        if (! options.fixedGenomePath.empty())
        {
            auto genome = readOneExternalGenome(options.fixedGenomePath, model.genomeSize());
            const float fixedLoss = evaluateGenome(model, genome, targets);
            std::cout << "Fixed genome loss: " << fixedLoss << "\n";
            metrics.logEvaluation(++evaluation, fixedLoss, std::min(initialLoss, fixedLoss), "fixed");
            metrics.logFinal(initialLoss, fixedLoss);
            writeResultJson(options.outputPath, model, genome, initialLoss, fixedLoss, options);
            exportExamples(options.exportDir, model, genome, targets);
            return 0;
        }

        Candidate best { initial, initialLoss };
        if (options.maxEvaluations > 1)
        {
            best = runPagmoPso(options, model, targets, initial, initialLoss, metrics, evaluation, bestLoss);
        }

        std::cout << "Best loss: " << best.loss << "\n";
        metrics.logFinal(initialLoss, best.loss);
        writeResultJson(options.outputPath, model, best.genome, initialLoss, best.loss, options);
        exportExamples(options.exportDir, model, best.genome, targets);
        return 0;
    }
    catch (const std::exception& e)
    {
        std::cerr << "PianoFit error: " << e.what() << "\n";
        printUsage(std::cerr);
        return 1;
    }
}
