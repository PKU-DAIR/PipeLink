#pragma once
#include <charconv>
#include <cmath>
#include <cstdint>
#include <limits>
#include <map>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace pipelink::attention::slot {
struct Json {
  enum class Kind { object, array, string, number, boolean, null };
  Kind kind = Kind::null;
  std::string text;
  std::string number_text;  // owned: parsed values outlive the parser's input buffer
  bool boolean_value = false;
  std::map<std::string, Json, std::less<>> members;
  std::vector<Json> elements;

  Json const& at(std::string_view key) const {
    if (kind != Kind::object) throw std::runtime_error("expected JSON object");
    auto member = members.find(key);
    if (member == members.end()) throw std::runtime_error("missing JSON member: " + std::string(key));
    return member->second;
  }

  std::string const& string() const {
    if (kind != Kind::string) throw std::runtime_error("expected JSON string");
    return text;
  }

  bool boolean() const {
    if (kind != Kind::boolean) throw std::runtime_error("expected JSON boolean");
    return boolean_value;
  }

  double number() const {
    if (kind != Kind::number) throw std::runtime_error("expected JSON number");
    double value;
    auto result = std::from_chars(number_text.data(), number_text.data() + number_text.size(), value);
    if (result.ec != std::errc{} || result.ptr != number_text.data() + number_text.size() || !std::isfinite(value)) {
      throw std::runtime_error("JSON number is outside the finite numeric range");
    }
    return value;
  }

  int integer() const {
    double value = number();
    if (std::trunc(value) != value || value < std::numeric_limits<int>::min() || value > std::numeric_limits<int>::max()) {
      throw std::runtime_error("expected JSON integer in int range");
    }
    return int(value);
  }

  std::vector<Json> const& array() const {
    if (kind != Kind::array) throw std::runtime_error("expected JSON array");
    return elements;
  }
};

class JsonParser {
 public:
  explicit JsonParser(std::string_view text) : text_(text) {}

  Json parse() {
    Json result = value();
    space();
    if (cursor_ != text_.size()) fail("trailing content");
    return result;
  }

 private:
  std::string_view text_;
  std::size_t cursor_ = 0;

  [[noreturn]] void fail(char const* reason) const {
    throw std::runtime_error("invalid JSON at byte " + std::to_string(cursor_) + ": " + reason);
  }

  char peek() const { return cursor_ < text_.size() ? text_[cursor_] : '\0'; }

  bool take(char c) {
    if (cursor_ == text_.size() || text_[cursor_] != c) return false;
    ++cursor_;
    return true;
  }

  void expect(char c) {
    if (!take(c)) fail("unexpected character");
  }

  void space() {
    while (peek() == ' ' || peek() == '\t' || peek() == '\r' || peek() == '\n') ++cursor_;
  }

  static bool digit(char c) { return c >= '0' && c <= '9'; }

  uint32_t hex4() {
    uint32_t result = 0;
    for (int i = 0; i < 4; ++i) {
      char c = peek();
      uint32_t nibble;
      if (c >= '0' && c <= '9') nibble = c - '0';
      else if (c >= 'a' && c <= 'f') nibble = c - 'a' + 10;
      else if (c >= 'A' && c <= 'F') nibble = c - 'A' + 10;
      else fail("invalid Unicode escape");
      ++cursor_;
      result = (result << 4) | nibble;
    }
    return result;
  }

  static void utf8(std::string& out, uint32_t codepoint) {
    if (codepoint <= 0x7f) out.push_back(char(codepoint));
    else {
      if (codepoint <= 0x7ff) out.push_back(char(0xc0 | (codepoint >> 6)));
      else {
        if (codepoint <= 0xffff) out.push_back(char(0xe0 | (codepoint >> 12)));
        else {
          out.push_back(char(0xf0 | (codepoint >> 18)));
          out.push_back(char(0x80 | ((codepoint >> 12) & 0x3f)));
        }
        out.push_back(char(0x80 | ((codepoint >> 6) & 0x3f)));
      }
      out.push_back(char(0x80 | (codepoint & 0x3f)));
    }
  }

  std::string string() {
    expect('"');
    std::string result;
    while (cursor_ < text_.size()) {
      unsigned char c = static_cast<unsigned char>(text_[cursor_++]);
      if (c == '"') return result;
      if (c < 0x20) fail("unescaped string control character");
      if (c == '\\') {
        if (cursor_ == text_.size()) fail("unterminated escape");
        switch (text_[cursor_++]) {
          case '"': result.push_back('"'); break;
          case '\\': result.push_back('\\'); break;
          case '/': result.push_back('/'); break;
          case 'b': result.push_back('\b'); break;
          case 'f': result.push_back('\f'); break;
          case 'n': result.push_back('\n'); break;
          case 'r': result.push_back('\r'); break;
          case 't': result.push_back('\t'); break;
          case 'u': {
            uint32_t codepoint = hex4();
            if (codepoint >= 0xd800 && codepoint <= 0xdbff) {
              expect('\\');
              expect('u');
              uint32_t low = hex4();
              if (low < 0xdc00 || low > 0xdfff) fail("invalid Unicode surrogate pair");
              codepoint = 0x10000 + ((codepoint - 0xd800) << 10) + low - 0xdc00;
            } else if (codepoint >= 0xdc00 && codepoint <= 0xdfff) fail("unpaired Unicode surrogate");
            utf8(result, codepoint);
            break;
          }
          default: fail("invalid string escape");
        }
      } else if (c < 0x80) result.push_back(char(c));
      else {
        std::size_t begin = cursor_ - 1;
        int continuation;
        uint32_t codepoint;
        uint32_t minimum;
        if (c >= 0xc2 && c <= 0xdf) { continuation = 1; codepoint = c & 0x1f; minimum = 0x80; }
        else if (c >= 0xe0 && c <= 0xef) { continuation = 2; codepoint = c & 0x0f; minimum = 0x800; }
        else if (c >= 0xf0 && c <= 0xf4) { continuation = 3; codepoint = c & 7; minimum = 0x10000; }
        else fail("invalid UTF-8");
        for (int i = 0; i < continuation; ++i) {
          unsigned char next = static_cast<unsigned char>(peek());
          if (next < 0x80 || next > 0xbf) fail("invalid UTF-8 continuation");
          ++cursor_;
          codepoint = (codepoint << 6) | (next & 0x3f);
        }
        if (codepoint < minimum || codepoint > 0x10ffff || (codepoint >= 0xd800 && codepoint <= 0xdfff)) fail("invalid UTF-8 codepoint");
        result.append(text_.data() + begin, cursor_ - begin);
      }
    }
    fail("unterminated string");
  }

  Json value() {
    space();
    Json result;
    if (take('{')) {
      result.kind = Json::Kind::object;
      space();
      if (take('}')) return result;
      do {
        space();
        std::string key = string();
        space();
        expect(':');
        result.members.insert_or_assign(std::move(key), value());
        space();
        if (take('}')) return result;
        expect(',');
      } while (true);
    }
    if (take('[')) {
      result.kind = Json::Kind::array;
      space();
      if (take(']')) return result;
      do {
        result.elements.push_back(value());
        space();
        if (take(']')) return result;
        expect(',');
      } while (true);
    }
    if (peek() == '"') {
      result.kind = Json::Kind::string;
      result.text = string();
      return result;
    }
    for (std::string_view literal : {"true", "false", "null"}) {
      if (text_.substr(cursor_, literal.size()) == literal) {
        cursor_ += literal.size();
        result.kind = literal == "null" ? Json::Kind::null : Json::Kind::boolean;
        result.boolean_value = literal == "true";
        return result;
      }
    }
    std::size_t begin = cursor_;
    take('-');
    if (!take('0')) {
      if (peek() < '1' || peek() > '9') fail("expected value");
      do { ++cursor_; } while (digit(peek()));
    }
    if (take('.')) {
      if (!digit(peek())) fail("missing fractional digits");
      do { ++cursor_; } while (digit(peek()));
    }
    if (take('e') || take('E')) {
      if (!take('+')) take('-');
      if (!digit(peek())) fail("missing exponent digits");
      do { ++cursor_; } while (digit(peek()));
    }
    result.kind = Json::Kind::number;
    result.number_text = std::string(text_.substr(begin, cursor_ - begin));
    return result;
  }
};

std::string json_quote(std::string_view value) {
  std::string result = "\"";
  for (unsigned char c : value) {
    if (c == '"' || c == '\\') {
      result.push_back('\\');
      result.push_back(char(c));
    } else if (c < 0x20) {
      constexpr char hex[] = "0123456789abcdef";
      result += "\\u00";
      result.push_back(hex[c >> 4]);
      result.push_back(hex[c & 15]);
    } else result.push_back(char(c));
  }
  result.push_back('"');
  return result;
}

} // namespace pipelink::attention::slot
