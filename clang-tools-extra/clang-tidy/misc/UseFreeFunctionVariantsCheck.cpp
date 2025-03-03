//===--- UseFreeFunctionVariantsCheck.cpp - clang-tidy --------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "UseFreeFunctionVariantsCheck.h"
#include "clang/AST/ASTContext.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/ASTMatchers/ASTMatchers.h"
#include "clang/Basic/SourceManager.h"
#include "clang/Lex/Lexer.h"
#include <iostream>

using namespace clang::ast_matchers;

namespace clang::tidy::misc {

void UseFreeFunctionVariantsCheck::registerMatchers(MatchFinder *Finder) {
    Finder->addMatcher(cxxMemberCallExpr(on(expr().bind("base")),
                                         callee(cxxMethodDecl(hasAnyName("isa", "dyn_cast", "cast")))
                ).bind("castCall"),
                       this);
}

// TODO(askrebko): Add checker for types wrapped in pointers and optional
void UseFreeFunctionVariantsCheck::check(
    const MatchFinder::MatchResult &Result) {
    const auto* CallExpr = Result.Nodes.getNodeAs<CXXMemberCallExpr>("castCall");
    const auto* BaseExpr = Result.Nodes.getNodeAs<Expr>("base");

    if (!CallExpr || !BaseExpr) {
        return;
    }
    SourceManager &SM = *Result.SourceManager;
    LangOptions LangOpts = getLangOpts();

    std::string BaseStr = Lexer::getSourceText(
        CharSourceRange::getTokenRange(BaseExpr->getSourceRange()), SM, LangOpts).str();

    const auto* MethodDecl = CallExpr->getMethodDecl();
    if (!MethodDecl->getTemplateSpecializationArgs()) {
        return;
    }

    std::string TemplateArgStr;
    for (const auto& Arg : MethodDecl->getTemplateSpecializationArgs()->asArray()) {
        if (Arg.getKind() == TemplateArgument::ArgKind::Type) {
            PrintingPolicy Policy(LangOpts);
            Policy.adjustForCPlusPlus();
            if (!TemplateArgStr.empty()) {
                TemplateArgStr += ", ";
            }
            TemplateArgStr += Arg.getAsType().getAsString(Policy);
        } else if (Arg.getKind() == TemplateArgument::ArgKind::Pack) {
            for (const auto& PackArg : Arg.pack_elements()) {
                PrintingPolicy Policy(LangOpts);
                Policy.adjustForCPlusPlus();
                if (!TemplateArgStr.empty()) {
                    TemplateArgStr += ", ";
                }
                TemplateArgStr += PackArg.getAsType().getAsString(Policy);
            }
        } else {
            std::cout << "Unsupported template argument kind: " << Arg.getKind() << std::endl;
            return;
        }
    }

    std::string FuncName = MethodDecl->getNameAsString();

    std::string Replacement = "mlir::" + FuncName + "<" + TemplateArgStr + ">(" + BaseStr + ")";
    diag(CallExpr->getBeginLoc(), "Replace 'type.isa<T>()' with 'mlir::isa<T>(type)'")
        << FixItHint::CreateReplacement(CallExpr->getSourceRange(), Replacement);
}

} // namespace clang::tidy::misc
