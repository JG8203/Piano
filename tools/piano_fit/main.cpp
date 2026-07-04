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
#include <numeric>
#include <optional>
#include <random>
#include <regex>
#include <sstream>
#include <string>
#include <unordered_set>
#include <vector>

extern Param params[NumParams];

namespace
{
constexpr int kSampleRate = 48000;
constexpr int kBlockSize = 512;
constexpr double kPi = 3.1415926535897932384626433832795;
constexpr float kEpsilon = 1.0e-7f;

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
    float sigma = 0.15f;
    float maxSeconds = 6.0f;
    uint32_t seed = 12345;
    bool printModelInfo = false;
    bool serveJsonl = false;
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
        << "  --eval-genomes <file>  Evaluate external genome JSONL instead of running built-in CMA-ES\n"
        << "  --eval-output <file>   Output JSONL losses for --eval-genomes\n"
        << "  --serve-jsonl          Serve JSONL evaluator requests on stdin/stdout\n"
        << "  --fixed-genome <file>  Evaluate/export one external genome JSON/JSONL instead of optimizing\n"
        << "  --subset pilot|all     Dataset subset (default pilot)\n"
        << "  --max-evals <n>        Evaluation budget; 0 evaluates defaults only\n"
        << "  --population <n>       CMA-ES population (default 40)\n"
        << "  --sigma <x>            Initial sigma (default 0.15)\n"
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
            options.sigma = std::stof(requireValue("--sigma"));
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
            value = std::clamp(value, 0.0f, 1.0f);
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
    return static_cast<float>(total / std::max<size_t>(1, targets.size()));
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
        if (! enabled)
            return;

        out << "{\"type\":\"config\""
            << ",\"subset\":\"" << jsonEscape(options.subset) << "\""
            << ",\"target_count\":" << targetCount
            << ",\"genome_size\":" << genomeSize
            << ",\"population\":" << options.population
            << ",\"sigma\":" << options.sigma
            << ",\"max_evaluations\":" << options.maxEvaluations
            << ",\"max_seconds\":" << options.maxSeconds
            << ",\"seed\":" << options.seed
            << "}\n";
        out.flush();
    }

    void logEvaluation(int evaluation, float loss, float bestLoss, const char* phase)
    {
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

    void logGeneration(int evaluations, int generation, float bestLoss, float sigma)
    {
        if (! enabled)
            return;

        out << std::setprecision(9)
            << "{\"type\":\"generation\""
            << ",\"evaluation\":" << evaluations
            << ",\"generation\":" << generation
            << ",\"best_loss\":" << bestLoss
            << ",\"sigma\":" << sigma
            << "}\n";
        out.flush();
    }

    void logFinal(float initialLoss, float bestLoss)
    {
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

class SeparableCmaEs
{
public:
    SeparableCmaEs(std::vector<float> initialMean, int population, float sigma, uint32_t seed)
        : mean(std::move(initialMean)),
          diagonalStd(mean.size(), 1.0f),
          populationSize(population),
          sigmaValue(sigma),
          rng(seed),
          normal(0.0f, 1.0f)
    {
        mu = std::max(2, populationSize / 2);
        weights.resize(static_cast<size_t>(mu));
        for (int i = 0; i < mu; ++i)
            weights[static_cast<size_t>(i)] = std::log((mu + 0.5) / (i + 1.0));
        const float sum = std::accumulate(weights.begin(), weights.end(), 0.0f);
        for (float& weight : weights)
            weight /= sum;
    }

    template <typename Fn, typename GenerationFn>
    Candidate run(int maxEvaluations, Candidate best, Fn&& evaluate, GenerationFn&& onGeneration)
    {
        int evaluations = 1;
        int generation = 0;

        while (evaluations < maxEvaluations)
        {
            std::vector<Candidate> candidates;
            candidates.reserve(static_cast<size_t>(populationSize));

            for (int i = 0; i < populationSize && evaluations < maxEvaluations; ++i)
            {
                Candidate candidate;
                candidate.genome.resize(mean.size());
                for (size_t d = 0; d < mean.size(); ++d)
                {
                    const float sample = mean[d] + sigmaValue * diagonalStd[d] * normal(rng);
                    candidate.genome[d] = std::clamp(sample, 0.0f, 1.0f);
                }
                candidate.loss = evaluate(candidate.genome);
                ++evaluations;
                if (candidate.loss < best.loss)
                    best = candidate;
                candidates.push_back(std::move(candidate));
            }

            std::sort(candidates.begin(), candidates.end(), [](const Candidate& a, const Candidate& b) { return a.loss < b.loss; });
            updateDistribution(candidates);
            ++generation;

            if (generation % 5 == 0 || generation == 1)
            {
                std::cout << "evals=" << evaluations
                          << " generation=" << generation
                          << " best=" << best.loss
                          << " sigma=" << sigmaValue << "\n";
            }
            onGeneration(evaluations, generation, best.loss, sigmaValue);
        }

        return best;
    }

private:
    void updateDistribution(const std::vector<Candidate>& candidates)
    {
        std::vector<float> newMean(mean.size(), 0.0f);
        for (int i = 0; i < mu && i < static_cast<int>(candidates.size()); ++i)
        {
            for (size_t d = 0; d < mean.size(); ++d)
                newMean[d] += weights[static_cast<size_t>(i)] * candidates[static_cast<size_t>(i)].genome[d];
        }

        std::vector<float> variance(mean.size(), 1.0e-4f);
        for (int i = 0; i < mu && i < static_cast<int>(candidates.size()); ++i)
        {
            for (size_t d = 0; d < mean.size(); ++d)
            {
                const float normalized = (candidates[static_cast<size_t>(i)].genome[d] - mean[d]) / std::max(sigmaValue, 1.0e-6f);
                variance[d] += weights[static_cast<size_t>(i)] * normalized * normalized;
            }
        }

        for (size_t d = 0; d < mean.size(); ++d)
        {
            mean[d] = std::clamp(newMean[d], 0.0f, 1.0f);
            diagonalStd[d] = std::clamp(0.85f * diagonalStd[d] + 0.15f * std::sqrt(variance[d]), 0.05f, 3.0f);
        }

        sigmaValue = std::max(0.015f, sigmaValue * 0.995f);
    }

    std::vector<float> mean;
    std::vector<float> diagonalStd;
    int populationSize = 40;
    int mu = 20;
    float sigmaValue = 0.15f;
    std::vector<float> weights;
    std::mt19937 rng;
    std::normal_distribution<float> normal;
};

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
    out << "  \"subset\": \"" << jsonEscape(options.subset) << "\",\n";
    out << "  \"population\": " << options.population << ",\n";
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
            SeparableCmaEs optimizer(initial, options.population, options.sigma, options.seed);
            best = optimizer.run(
                options.maxEvaluations,
                best,
                [&](const std::vector<float>& genome) {
                    const float loss = evaluateGenome(model, genome, targets);
                    ++evaluation;
                    bestLoss = std::min(bestLoss, loss);
                    metrics.logEvaluation(evaluation, loss, bestLoss, "candidate");
                    return loss;
                },
                [&](int evaluations, int generation, float generationBestLoss, float sigma) {
                    metrics.logGeneration(evaluations, generation, generationBestLoss, sigma);
                });
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
