# What is actually in this repository

**Measured, not assumed.** Five files: four legal/attribution documents and a
README. No Python, no tests, no build.

```
ATTRIBUTION.md  LICENSE  LICENSE-COMMERCIAL.md  NOTICE.md  README.md
```

## The README is a reconstruction document, not a description

It opens honestly — *"I don't know how to do this, so that's your problem
now"* — and then carries a full technical audit describing **~20,000 lines of
code and ~15,000 lines of documentation** for ML Filesystem v1.8+.

**None of that code is here.** The audit is the artifact; the system it audits
is not.

That gap is worth stating plainly, because a reader arriving at a repository
named `ML_Filesystem_0.95` reasonably expects an ML filesystem, and a model
reading it will summarise the audit as though it were describing the contents.

## Where the system actually lives

The implementation is in the capsules, not in this repository:

    ml-filesystem-monolith          74 source files
    ml-filesystem-v18-enhancement   the 8 enhancements and Enhanced Agent

Both are held and indexed. In the Tower's holdings `ml-filesystem` is the
single largest documentary Spire at **285 units** — and essentially all of it
comes from those capsules rather than from here.

## What this repository needs

Nothing to build. A repo that is a specification plus its licences is a
legitimate thing to be, and inventing tests or tooling for five documents
would produce something nobody maintains.

What it needed was for the gap between its name and its contents to be
written down once, so the next reader does not have to work it out.

—Shibbieness
—Claude
