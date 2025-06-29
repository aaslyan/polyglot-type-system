#pragma once

#include <vector>
#include <string>
#include <memory>
#include <map>

namespace utils {

template<typename T>
class VectorWrapper {
private:
    std::vector<T> data_;
    mutable size_t access_count_;

public:
    VectorWrapper() : access_count_(0) {}
    explicit VectorWrapper(size_t size) : data_(size), access_count_(0) {}

    void push_back(const T& value) {
        data_.push_back(value);
    }

    T& operator[](size_t index) {
        ++access_count_;
        return data_[index];
    }

    const T& operator[](size_t index) const {
        ++access_count_;
        return data_[index];
    }

    size_t size() const noexcept {
        return data_.size();
    }

    bool empty() const noexcept {
        return data_.empty();
    }

    size_t get_access_count() const noexcept {
        return access_count_;
    }
};

struct Point3D {
    double x, y, z;

    Point3D() : x(0), y(0), z(0) {}
    Point3D(double x_, double y_, double z_) : x(x_), y(y_), z(z_) {}

    double magnitude() const noexcept;
    Point3D normalize() const;

    Point3D operator+(const Point3D& other) const {
        return Point3D(x + other.x, y + other.y, z + other.z);
    }
};

class DataProcessor {
public:
    using ResultMap = std::map<std::string, std::vector<double>>;

    virtual ~DataProcessor() = default;

    virtual ResultMap process(const std::vector<Point3D>& points) = 0;
    virtual void reset() noexcept = 0;

protected:
    std::unique_ptr<VectorWrapper<double>> cache_;
};

// Function declarations
std::vector<int> parse_integers(const std::string& input);
std::unique_ptr<DataProcessor> create_processor(const std::string& type);

template<typename T, typename U>
std::vector<std::pair<T, U>> zip_vectors(const std::vector<T>& v1, 
                                         const std::vector<U>& v2);

} // namespace utils
