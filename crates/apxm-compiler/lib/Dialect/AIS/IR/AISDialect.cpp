/**
 * @file AISDialect.cpp
 * @brief MLIR dialect implementation for the AIS dialect.
 *
 * This file register the AIS dialect, its custom types (Token, Handle,
 * Goal), attributes, and the associated parser/printer logic.
 */

#include "apxm/Dialect/AIS/IR/AISDialect.h"
#include "apxm/Dialect/AIS/IR/AISAttributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;
using namespace mlir::ais;

#include "apxm/Dialect/AIS/IR/AISEnums.cpp.inc"
#define GET_ATTRDEF_CLASSES
#include "apxm/Dialect/AIS/IR/AISAttributes.cpp.inc"

void AISDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "apxm/Dialect/AIS/IR/AISOps.cpp.inc"
      >();
  addAttributes<
#define GET_ATTRDEF_LIST
#include "apxm/Dialect/AIS/IR/AISAttributes.cpp.inc"
      >();
  addTypes<TokenType, HandleType, GoalType>();
}

Type AISDialect::parseType(DialectAsmParser &parser) const {

  MLIRContext *ctx = getContext();
  StringRef keyword;

  if (failed(parser.parseKeyword(&keyword)))
    return Type();

  auto parseToken = [&]() -> Type {
    Type payload = getDefaultPayloadType(ctx);
    if (succeeded(parser.parseOptionalLess())) {
      if (parser.parseType(payload) || parser.parseGreater())
        return Type();
    }
    return TokenType::get(ctx, payload);
  };

  auto parseHandle = [&]() -> Type {
    if (parser.parseLess())
      return Type();

    StringRef spaceKeyword;
    if (parser.parseKeyword(&spaceKeyword))
      return Type();

    auto space = symbolizeMemorySpace(spaceKeyword);
    if (!space) {
      parser.emitError(parser.getCurrentLocation())
          << "unknown AIS memory space \"" << spaceKeyword << "\"";
      return Type();
    }

    Type payload = getDefaultPayloadType(ctx);
    if (succeeded(parser.parseOptionalComma()) && parser.parseType(payload))
      return Type();

    if (parser.parseGreater())
      return Type();

    return HandleType::get(ctx, *space, payload);
  };

  auto parseGoal = [&]() -> Type {
    unsigned priority = 0;
    if (succeeded(parser.parseOptionalLess())) {
      uint64_t parsedPriority = 0;
      if (parser.parseInteger(parsedPriority) || parser.parseGreater())
        return Type();
      priority = static_cast<unsigned>(parsedPriority);
    }
    return GoalType::get(ctx, priority);
  };

  if (keyword == "token")
    return parseToken();
  if (keyword == "handle")
    return parseHandle();
  if (keyword == "goal")
    return parseGoal();

  parser.emitError(parser.getCurrentLocation()) << "unknown AIS type \"" << keyword << "\"";
  return Type();
}

void AISDialect::printType(Type type, DialectAsmPrinter &printer) const {
  TypeSwitch<Type>(type)
      .Case<TokenType>([&](TokenType token) {
        printer << "token";
        Type payload = token.getInnerType();
        if (!llvm::isa<NoneType>(payload)) {
          printer << '<';
          printer.printType(payload);
          printer << '>';
        }
      })
      .Case<HandleType>([&](HandleType handle) {
        printer << "handle<" << stringifyMemorySpace(handle.getSpace());
        Type payload = handle.getPayload();
        if (!llvm::isa<NoneType>(payload)) {
          printer << ", ";
          printer.printType(payload);
        }
        printer << '>';
      })
      .Case<GoalType>([&](GoalType goal) { printer << "goal<" << goal.getPriority() << '>'; })
      .Default([](Type) { llvm_unreachable("unknown AIS type"); });
}

#include "apxm/Dialect/AIS/IR/AISDialect.cpp.inc"
