/*
   MIT License

   Copyright (c) 2017-2018 Matthias C. M. Troffaes

   Permission is hereby granted, free of charge, to any person obtaining a copy
   of this software and associated documentation files (the "Software"), to deal
   in the Software without restriction, including without limitation the rights
   to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
   copies of the Software, and to permit persons to whom the Software is
   furnished to do so, subject to the following conditions:

   The above copyright notice and this permission notice shall be included in all
   copies or substantial portions of the Software.

   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
   IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
   FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
   AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
   LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
   OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
   SOFTWARE.
 */

#ifndef INIPARSER_H
#define INIPARSER_H

#pragma once

#include <cstring>
#include <string>
#include <iostream>
#include <list>
#include <map>
#include <algorithm>
#include <functional>
#include <cctype>
#include <sstream>
#include <vector>

namespace inipp
{

class Ini
{
public:
  using String = std::string;
  using Section = std::map<String, String>;
  using Sections = std::map<String, Section>;
  using Symbols = std::list<std::pair<String, String> >;

  Sections sections;
  std::list<String> errors;

  static const char char_section_start  = '[';
  static const char char_section_end    = ']';
  static const char char_assign         = '=';
  static const char char_comment        = '#';
  static const char char_interpol       = '$';
  static const char char_interpol_start = '{';
  static const char char_interpol_sep   = ':';
  static const char char_interpol_end   = '}';

  static const int max_interpolation_depth = 10;

  template <typename TT>
  bool extract(const String & key,
               const String & value,
               TT & dst)
  {
    auto lowerCaseKey = toLowerCase(key);
    auto lowerCaseValue = toLowerCase(value);
    if (sections[lowerCaseKey].count(lowerCaseValue)) {
      if (std::is_same<TT, bool>::value) {
        dst = string_to_boolean(sections[lowerCaseKey][lowerCaseValue]);
      }
      else {
        std::istringstream is{sections[lowerCaseKey][lowerCaseValue]};
        is >> dst;
      }
      return true;
    }
    return false;
  }

  template <typename TT> bool set(const String &key, const String &value, TT &&src)
  {
    auto lowerCaseKey = toLowerCase(key);
    auto lowerCaseValue = toLowerCase(value);
    if (sections[lowerCaseKey].count(lowerCaseValue)) {
      if (std::is_same<TT, bool>::value) {
        sections[lowerCaseKey][lowerCaseValue] = src;
      }
      else {
        std::ostringstream ss;
        ss << src;
        sections[lowerCaseKey][lowerCaseValue] = ss.str();
      }
      return true;
    }
    return false;
  }

  bool extract(const String & key,
               const String & value,
               String & dst) ;

  void generate(std::ostringstream & os) const;

  void parse(std::stringstream & is, bool lowerValue = true);

  void interpolate();

  void default_section(const Section & sec);

  void clear();

private:
  enum string_to_boolean_t { boolean_false, boolean_true, boolean_invalid };

  String local_symbol(const String & name) const;

  String global_symbol(const String & sec_name, const String & name) const;

  Symbols local_symbols(const String & sec_name, const Section & sec) const;

  Symbols global_symbols() const;

  bool replace_symbols(const Symbols & syms, Section & sec) const;

  static String toLowerCase(const String &input)
  {
    String output(input);
    std::transform(output.begin(), output.end(), output.begin(), [](auto c) { return std::tolower(c); });
    return output;
  }
  static string_to_boolean_t string_to_boolean(const std::string s, bool strict = false)
  {
    const char *falses[] = {"false", "no", "0"};
    const char *trues[] = {"true", "yes", "1"};

    unsigned num_falses = sizeof(falses) / sizeof(falses[0]);
    unsigned num_trues = sizeof(trues) / sizeof(trues[0]);

    // Special case
    if (s.empty())
      return boolean_invalid;

    // Get a lowercase version of 's'
    std::string s2(s);
    std::transform(s2.begin(), s2.end(), s2.begin(), [](int c) { return std::tolower(c); });

    // Does the string represent a FALSE?
    for (unsigned n = 0; n < num_falses; n++)
      if (std::string(falses[n]).find(s2) == 0)
        return boolean_false;

    // Does the string represent a TRUE?
    for (unsigned n = 0; n < num_trues; n++)
      if (std::string(trues[n]).find(s2) == 0)
        return boolean_true;

    // Check for non-zero numbers here
    if (!strict) {
      std::istringstream ss(s2);
      double d;
      if (ss >> d)
        return (d == 0.0) ? boolean_false : boolean_true;
    }

    // The string was not recognized
    return boolean_invalid;
  }
};
} // namespace inipp

#endif
