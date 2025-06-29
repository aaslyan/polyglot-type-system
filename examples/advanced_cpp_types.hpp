#pragma once

#include <memory>
#include <vector>
#include <functional>
#include <tuple>
#include <type_traits>
#include <concepts>
#include <optional>
#include <variant>

// Variadic template example
template<typename... Args>
class VariadicContainer {
public:
    std::tuple<Args...> data;
    
    template<typename T>
    void process(T&& value) {
        // Process each type
    }
    
    constexpr size_t size() const { return sizeof...(Args); }
};

// SFINAE example
template<typename T, typename = void>
struct has_iterator : std::false_type {};

template<typename T>
struct has_iterator<T, std::void_t<
    typename T::iterator,
    typename T::const_iterator
>> : std::true_type {};

// C++20 Concepts example
template<typename T>
concept Arithmetic = std::is_arithmetic_v<T>;

template<typename T>
concept Container = requires(T t) {
    typename T::value_type;
    typename T::size_type;
    { t.size() } -> std::convertible_to<std::size_t>;
    { t.begin() } -> std::same_as<typename T::iterator>;
};

template<Arithmetic T>
class MathVector {
    std::vector<T> data;
public:
    T dot_product(const MathVector<T>& other) const;
};

// Complex inheritance hierarchy
class Shape {
public:
    virtual double area() const = 0;
    virtual ~Shape() = default;
};

class Drawable {
public:
    virtual void draw() const = 0;
    virtual ~Drawable() = default;
};

class Circle : public Shape, public Drawable {
    double radius;
public:
    explicit Circle(double r) : radius(r) {}
    double area() const override { return 3.14159 * radius * radius; }
    void draw() const override { /* drawing logic */ }
};

// Function pointers and std::function
using SimpleCallback = void(*)(int);
using ComplexCallback = std::function<int(const std::string&, double)>;

class EventHandler {
    std::vector<ComplexCallback> callbacks;
public:
    void register_callback(ComplexCallback cb) {
        callbacks.push_back(std::move(cb));
    }
};

// Advanced template with multiple parameters and constraints
template<typename T, size_t N, typename Allocator = std::allocator<T>>
    requires std::is_default_constructible_v<T>
class FixedArray {
    T data[N];
    Allocator alloc;
public:
    using value_type = T;
    using size_type = size_t;
    constexpr size_type size() const noexcept { return N; }
};

// Nested templates and type aliases
template<typename Key, typename Value>
class Cache {
public:
    using key_type = Key;
    using value_type = Value;
    using pair_type = std::pair<Key, Value>;
    
    template<typename K, typename V>
    struct Node {
        K key;
        V value;
        std::shared_ptr<Node<K, V>> next;
    };
    
    using node_type = Node<Key, Value>;
    using node_ptr = std::shared_ptr<node_type>;
};

// std::variant and std::optional usage
using ConfigValue = std::variant<int, double, std::string, bool>;

struct Configuration {
    std::optional<std::string> name;
    std::vector<ConfigValue> values;
    
    template<typename T>
    std::optional<T> get_value(size_t index) const {
        if (index >= values.size()) return std::nullopt;
        
        if (auto* val = std::get_if<T>(&values[index])) {
            return *val;
        }
        return std::nullopt;
    }
};

// Reference wrapper and perfect forwarding
template<typename T>
class SmartWrapper {
    std::reference_wrapper<T> ref;
public:
    explicit SmartWrapper(T& obj) : ref(obj) {}
    
    template<typename... Args>
    auto call(Args&&... args) -> decltype(ref.get()(std::forward<Args>(args)...)) {
        return ref.get()(std::forward<Args>(args)...);
    }
};

// Structured bindings support
struct Point3D {
    double x, y, z;
    
    auto operator<=>(const Point3D&) const = default;
    
    // Enable structured bindings
    template<std::size_t I>
    auto get() const {
        if constexpr (I == 0) return x;
        else if constexpr (I == 1) return y;
        else if constexpr (I == 2) return z;
    }
};

// Specialization for std::tuple_size and std::tuple_element
namespace std {
    template<>
    struct tuple_size<Point3D> : integral_constant<size_t, 3> {};
    
    template<size_t I>
    struct tuple_element<I, Point3D> {
        using type = double;
    };
}