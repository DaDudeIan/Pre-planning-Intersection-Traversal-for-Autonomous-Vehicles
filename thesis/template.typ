// This function gets your whole document as its `body` and formats
// it as an article in the style of the IEEE.
#import "lib/mod.typ": *

#let appendix = state("appendix", false)
#let heading-supplement = state("heading-supplement", "Section")

#let start-appendix(
  show-title: true,
  show-toc: true,
) = {
  counter(heading).update(0)
  appendix.update(true)
  heading-supplement.update("Appendix")
  if show-title {
    [
      #heading(
        level: 1,
        numbering: none,
        supplement: heading-supplement.get(),
        [Appendix]
      )<appendix>
    ]
  }

  if show-toc {
    // [#heading([Contents], level: 2, numbering: none, outlined: false)<contents-1>]
    outline(
      title: none,
      indent: auto,
      depth: 2,
      target: heading.where(numbering: "A.1:"),
    )
  }
}

#let paper(
    // The paper's title.
    title: "Paper Title",
    subtitle: none,

    title-page: false,
    title-page-extra: none,
    title-page-footer: none,
    title-page-content: none,

    // An array of authors. For each author you can specify a name,
    // department, organization, location, and email. Everything but
    // but the name is optional.
    authors: (),

    // The paper's publication date. Can be omitted if you don't have
    // one.
    date: none,

    // The paper's abstract. Can be omitted if you don't have one.
    abstract: none,

    // A list of index terms to display after the abstract.
    index-terms: (),

    // The article's paper size. Also affects the margins.
    paper-size: "a4",
    two-column: false,

    // colors
    accent: blue,

    // The path to a bibliography file if you want to cite some external
    // works.
    bibliography-file: none,

    // Whether to print a table of contents.
    print-toc: false,
    toc-depth: none,
    toc-printer: (depth) => {
        set par(first-line-indent: 0em)
        outline(
            indent: true,
            fill: grid(
                columns: 1,
                block(
                    fill: black,
                    height: 0.5pt,
                    width: 100%,
                ),
                block(
                    fill: none,
                    height: 0.25em,
                    width: 100%,
                ),
            ),
            depth: depth,
        )
        // pagebreak(weak: true)
        colbreak(weak: true)
    },

    prebody: none,
    postbody: none,

    // The paper's content.
    body
) = {
    // Set document metadata.
    set document(title: title, author: authors.map(author => author.name))

    // Set the body font.
    set text(font: "STIX Two Text", size: 12pt)
    // set text(font: "New Computer Modern", size: 12pt)

    // Configure the page.
    set page(
        // fill: theme.base,
        paper: paper-size,
        numbering: "i", // "1/1"
        number-align: center,
    )

    show cite: citation => {
        show regex("\d+"): set text(accent)
        [#citation]
    }

    show math.equation.where(block: true) : set block(spacing: 1.25em)

    set math.mat(delim: "[")

    show figure: it => align(center)[
        #let fs = 0.8em
        #let bh = 2em
        #if it.kind == "algorithm" {
            fs = 1em
            bh = 0em
        }
        #set text(size: fs)
        #it.body
        #v(1em, weak: true)

        #if it.kind != "algorithm" [
            #set text(accent)
            #it.supplement
            #it.counter.display(it.numbering).
            #set text(black)
            #if it.caption != none {
                it.caption.body
            }
        ] #h(0.1em)
        // #it.caption
        // #repr(it.caption)
        #v(bh, weak: true)
    ]

    show figure.where(kind: "example") : it => {
        it.body
    }

    show figure.where(kind: "algorithm") : it => {
        it.body
    }

    // triggered when using the dedicated syntax `@ref`
    show ref: it => {
      let el = it.element

      // return repr(el)

      if el == none {
          it.citation
      }
      else {
        let eq = math.equation

        if el.func() == eq {
          // The reference is an equation
          let sup = if it.fields().at("supplement", default: "none") == "none" {
            [Equation]
          } else { [] }
          // let sup = it.fields().at("supplement", default: "none")
          // show link : set text(black)
          // show regex("\d+"): set text(accent)
          // let n = numbering(el.numbering, ..counter(eq).at(el.location()))
          let n = counter(eq).at(el.location()).at(0)
          return [#link(it.target, sup) \(#link(it.target, [#n])\)]
        }
        else if el.has("kind") and el.kind == "example" {
            let loc = it.element.location()
            let exs = query(selector(<meta:excounter>).after(loc), loc)
            let number = example-counter.at(exs.first().location())

            return link(
              it.target,
              [#el.supplement~#numbering(it.element.numbering, ..number)]
            )
        }
        else {
            return link(
              it.target,
              it
            )
        }
      }

    }

    // Configure equation numbering and spacing
    set math.equation(numbering: "(1)")
    // show math.equation: set block(spacing: 1.25em)

    // Configure lists
    set list(indent: 0em, body-indent: 0.5em)
    set enum(indent: 0em, body-indent: 0.5em)

    // Configure headings
    set heading(numbering: "1.1.1")

    show heading.where(level: 1) : set text (size: 24pt)
    show heading.where(level: 2) : set text (size: 18pt)
    show heading.where(level: 3) : set text (size: 16pt)
    show heading.where(level: 4) : set text (size: 14pt)

    // functions
    // display the mails in a set construction manner
    let display-mails() = {
        // make unique set of dicts all email domains and a list of their mail prefixes
        let mail_set = (:)
        for author in authors {
            if "email" in author {
                // text(author.email.split("@").join(", "))
                let mail = author.email.split("@").at(0)
                let domain = author.email.split("@").at(1)

                if domain in mail_set {
                    mail_set.at(domain).push(mail)
                } else {
                    mail_set.insert(domain, (mail,))
                }
            }
        }

        // display the set construction(s)
        // making sure to only do the set construction syntax if one domain includes multiple mails
        for domain in mail_set.keys() {
            let mails = mail_set.at(domain)

            if mails.len() > 1 {
                text("{" + mails.join(", ") + "}" + "@" + domain)
            } else {
                text(mails.at(0) + "@" + domain)
            }
            if mail_set.keys().last() != domain {
                text(", ")
            }
        }
    }

    // Find all unique affiliations
    let process-affiliations(authors) = {
        // make sure to filter out duplicate affiliations
        // make each affiliation a dictionary with three keys:
        // department, organization, location
        // then filter out duplicate dictionaries, where all
        // three key/values pairs are the same

        let affiliations = authors.map(author => {
            let meta = (:)
            if "name" in author {
                meta.insert("name", author.name)
            }
            let info = (:)
            if "department" in author {
                info.insert("department", author.department)
            }
            if "organization" in author {
                info.insert("organization", author.organization)
            }
            if "location" in author {
                info.insert("location", author.location)
            }
            let affiliation = (
                    meta: meta,
                    info: info
            )
            affiliation
        })

        // filter out duplicate affiliations

        let unique-affiliations = ()
        for affiliation in affiliations {
            let is-unique = true
            for other in unique-affiliations {
                if affiliation.info == other.info {
                    is-unique = false
                    break
                }
            }
            if is-unique {
                affiliation.meta.insert("index", unique-affiliations.len() + 1)

                // update the author's affiliation index
                authors = authors.map(author => {
                    let author_affiliation = (
                        department: author.department,
                        organization: author.organization,
                        location: author.location,
                    )
                    if author_affiliation == affiliation.info {
                        author.insert("affiliation", affiliation.meta.index)
                    }
                    author
                })

                unique-affiliations.push(affiliation)
            }
        }
        (unique-affiliations, authors)
    }

    let display-title() = {
        // Display the paper's title.
        text(18pt, title, weight: 600)
        if subtitle != none {
            v(5mm, weak: true)
            align(center, text(14pt, subtitle))
        }
    }

    let display-names(authors, affiliation-set) = {
        // display author names as comma separated list, the last two
        // separated by "and" with oxford comma.
        // including a superscripted affiliation index in front of each name
        set text(12pt, weight: 700)

        let names = authors.map(author => {
            let name = [#author.name]
            if affiliation-set.len() > 1 {
                let index = author.affiliation
                name = super[#index] + name
            }
            name
        })

        if names.len() == 1 {
            text(names.at(0))
        } else if names.len() == 2 {
            text(names.at(0) + " & " + names.at(1))
        } else {
            text(names.slice(0, -1).join(", ") + ", and " + names.at(-1))
        }
    }

    let display-affiliations(affiliation-set) = {
        // display the author affiliations
        // that is: department, organization, location
        // except for the email addresses

        set text(10pt, weight: 400)

        // display the affiliations as a grid
        // with one column per affiliation

        grid(
            columns: affiliation-set.len() * (1fr,),
            gutter: 12pt,
            ..affiliation-set.map(affiliation => {
                let a = affiliation.info
                let i = affiliation.meta.index
                if affiliation-set.len() > 1 [
                    #super[#i]
                ]
                if "department" in a [
                    #emph([#a.department])
                ]
                if "organization" in a [
                    \ #emph(a.organization)
                ]
                if "location" in a [
                    \ #a.location
                ]
            })
        )
    }

    // display the date
    let display-date() = {
        if date == none {
            return
        }
        else if date == "today" {
            let today = datetime.today()
            let day-suffix = "th"
            let modten = calc.rem(today.day(), 10)
            if modten == 1 {
                "st"
            } else if modten == 2 {
                "nd"
            } else if modten == 3 {
                "rd"
            }
            text(10pt, weight: 700, today.display("[month repr:long] [day]" + day-suffix + " [year]"))
        } else {
            text(10pt, weight: 700, date)
        }
    }

    let make-title(authors, unique-affiliations) = {
        set align(center)

        // Display the paper's title.
        v(3pt, weak: true)
        display-title()

        // display the author names
        v(8.35mm, weak: true)
        display-names(authors, unique-affiliations)

        // display the author affiliations
        display-affiliations(unique-affiliations)

        // display the email addresses
        v(0.75em, weak: true)
        display-mails()

        // display the date
        if (date != none) {
            v(5.65mm, weak: true)
            display-date()
        }
    }

    let make-titlepage(authors, unique-affiliations) = {
        // Title page
        set page(numbering: none)
        if (title-page-content != none) {
            title-page-content
        } else {
            make-title(authors, unique-affiliations)
            v(1fr, weak: true)
            title-page-extra
            v(1fr, weak: true)
            title-page-footer
        }
        pagebreak(weak: true)
    }

    let breaklink(url) = link(url, for c in url [#c.replace(c,c+sym.zws)]) // zws -> zero width space

    show regex("https?://[^\s]+"): url => {
        breaklink(url.text)
    }

    let (unique-affiliations, authors) = process-affiliations(authors)
    if title-page {
        make-titlepage(authors, unique-affiliations)
        counter(page).update(1)
    } else {
        make-title(authors, unique-affiliations)
    }

    // Paper contents
    // reset text settings
    set align(left)
    // set text(10pt, weight: 400)

    v(40pt, weak: true)

    // Start two column mode and configure paragraph properties.
    set par(justify: true, first-line-indent: 1em, spacing: 1.25em)

    // Print TOC if print-toc is true
    if print-toc {
        toc-printer(toc-depth)
    }

    {
        let col-amount = 1
        let col-gutter = 0pt
        if two-column {
            col-amount = 2
            col-gutter = 12pt
            show: columns.with(col-amount, gutter: col-gutter)
        }

        prebody

        // Display abstract and index terms.
        if abstract != none [
            #set text(weight: 700)
            #h(1em) _Abstract_---#abstract

            #if index-terms != () [
                #h(1em)_Index terms_---#index-terms.join(", ")
            ]
            #v(2pt)
        ]

        // Display the paper's contents.
        body

        let ref-text-size = if two-column { 10pt } else { 14pt }
        // Display bibliography.
        if bibliography-file != none {
            show bibliography: set text(8pt)
            // bibliography(bibliography-file, title: text(ref-text-size)[References], style: "ieee")
            heading(
                level: 1,
                numbering: none,
                supplement: "References",
                [References]
            )
            bibliography(bibliography-file, title: none, style: "ieee")
        }
    }

    postbody
}
