#include <IO/SWCIO.hpp>
#include <fstream>

VISER_BEGIN

    class SWCFilePrivate{
    public:

        void ReadFromTxtFile(){
            std::string str;
            std::stringstream ss;
            fs.seekg(0, std::ios::beg);
            ss << fs.rdbuf();
            str = std::move(ss.str());
            bool file_end = false;
            size_t pos = 0;
            size_t len = str.length();
            auto get_line = [&](){
                std::string line;
                while(pos < len && str[pos] != '\n'){
                    line += str[pos++];
                }
                while(pos < len && str[pos] == '\n') pos++;
                if(pos >= len) file_end = true;
                return line;
            };
            const char* delim = " ";
            //因为使用了 stoi和stof 会抛出异常
            auto parse = [](const auto& token, SWCFile::SWCPoint& point){
                if(token.size() != 7){
                    throw std::runtime_error("Invalid token size for parse SWCPoint : " + std::to_string(token.size()));
                }
                point.id     = std::stoi(token[0]);
                point.tag    = std::stoi(token[1]);
                point.x      = std::stof(token[2]);
                point.y      = std::stof(token[3]);
                point.z      = std::stof(token[4]);
                point.radius = std::stof(token[5]);
                point.pid    = std::stoi(token[6]);
            };
            while(!file_end){
                auto line = get_line();
                if(line.empty() || line.front() == '#'){
                    continue;
                }
                auto token = vutil::split(line, delim);
                auto& point = points.emplace_back();
                // 转换失败的会被抛弃 不会向外层抛出异常 但是会输出错误日志
                try{
                    parse(token, point);
                }
                catch (const std::exception& err) {
                    std::cerr << err.what() << std::endl;
                    points.pop_back();
                }
            }
        }

        void ReadFromBinaryFile(){
            // file size 必须是 sizeof(SWCPoint) 的整数倍
            fs.seekg(0, std::ios::end);
            size_t file_size = fs.tellg();
            fs.seekg(0, std::ios::beg);
            size_t count = file_size / sizeof(SWCFile::SWCPoint);
            if(count == 0){
                std::cerr << "Read binary SWCFile with zero size" << std::endl;
                return;
            }
            points.resize(count);
            fs.read(reinterpret_cast<char*>(points.data()), count * sizeof(SWCFile::SWCPoint));
        }

        void WritePointToTxt(const SWCFile::SWCPoint& point){
            auto to_string = [](const SWCFile::SWCPoint& point){
                std::string str;
                str += std::to_string(point.id) + " ";
                str += std::to_string(point.tag) + " ";
                str += std::to_string(point.x) + " ";
                str += std::to_string(point.y) + " ";
                str += std::to_string(point.z) + " ";
                str += std::to_string(point.radius) + " ";
                str += std::to_string(point.pid) + "\n";
                return str;
            };
            auto str = to_string(point);
            fs.seekp(0, std::ios::end);
            fs.write(str.data(), str.length());
        }
        void WritePointToBin(const SWCFile::SWCPoint& point){
            fs.seekp(0, std::ios::end);
            fs.write(reinterpret_cast<const char*>(&point), sizeof(point));
        }
        std::vector<SWCFile::SWCPoint> points;
        std::fstream fs;
        enum FileType{
            TXT,
            BIN
        };
        FileType file_type;
        std::mutex mtx;
        UnifiedRescUID uid;
    };

    SWCFile::SWCFile()
    :_(std::make_unique<SWCFilePrivate>())
    {
        _->uid = SWCIOInterface::GenUnifiedRescUID();
    }

    SWCFile::~SWCFile() {
        Close();
    }

    void SWCFile::Lock() {
        _->mtx.lock();
    }

    void SWCFile::UnLock() {
        _->mtx.unlock();
    }

    UnifiedRescUID SWCFile::GetUID() const {
        return _->uid;
    }

    void SWCFile::Open(std::string_view filename, Mode mode) {
        Close();
        if(mode == Write){
           if(vutil::ends_with(filename, SWC_FILENAME_EXT_TXT)){
               _->file_type = SWCFilePrivate::TXT;
               _->fs.open(filename.data(), std::ios::out);
           }
           else if(vutil::ends_with(filename, SWC_FILENAME_EXT_BIN)){
               _->file_type = SWCFilePrivate::BIN;
               _->fs.open(filename.data(), std::ios::out | std::ios::binary);
           }
           else{
               throw ViserFileOpenError("Invalid ext for open SWC file to write");
           }
        }
        else{
            if(vutil::ends_with(filename, SWC_FILENAME_EXT_TXT)){
                _->file_type = SWCFilePrivate::TXT;
                _->fs.open(filename.data(), std::ios::in);
            }
            else if(vutil::ends_with(filename, SWC_FILENAME_EXT_BIN)){
                _->file_type = SWCFilePrivate::BIN;
                _->fs.open(filename.data(), std::ios::in | std::ios::binary);
            }
            else{
                throw ViserFileOpenError("Invalid ext for open SWC file to read");
            }
        }
        if(!_->fs.is_open()){
            throw ViserFileOpenError("Failed to open SWC file : " + std::string(filename));
        }
    }

    std::vector<SWCFile::SWCPoint> SWCFile::GetAllPoints() noexcept {
        _->points.clear();
        try{
            if(_->file_type == SWCFilePrivate::TXT){
                _->ReadFromTxtFile();
            }
            else{
                _->ReadFromBinaryFile();
            }
        } catch (const std::exception& err) {
            std::cerr << "Get all points from swc file failed with : " << err.what() << std::endl;
            return {};
        }
        return std::move(_->points);
    }

    void SWCFile::WritePoints(int count, std::function<const SWCPoint &(int)> get) noexcept {
        _->points.clear();
        auto write = [&](std::function<void(const SWCPoint&)> write_func){
            for(int id = 1; id <= count; ++id){
                write_func(get(id));
            }
        };
        if(_->file_type == SWCFilePrivate::TXT){
            write([&](const SWCPoint& point){
                _->WritePointToTxt(point);
            });
        }
        else{
            write([&](const SWCPoint& point){
                _->WritePointToBin(point);
            });
        }
    }

    void SWCFile::WritePoints(const std::vector<SWCPoint>& points) noexcept {
        _->points.clear();
        auto write = [&](std::function<void(const SWCPoint&)> write_func){
            for(const auto& point : points){
                write_func(point);
            }
        };
        if(_->file_type == SWCFilePrivate::TXT){
            write([&](const SWCPoint& point){
                _->WritePointToTxt(point);
            });
        }
        else{
            write([&](const SWCPoint& point){
                _->WritePointToBin(point);
            });
        }
    }

    void SWCFile::Close() noexcept {
        if(_->fs.is_open())
            _->fs.close();
        _->points.clear();
    }

VISER_END