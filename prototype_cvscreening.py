from pydantic import BaseModel
from crewai import Agent, Task, Crew, Process, LLM
import json


VERBOSE = False
TRACING = True

# Model Initialisation
llm = LLM(
    model="ollama/granite4.1:3b",
    base_url="http://localhost:11434",
    temperature=0.1,
)

# ======================================================================
# STEP 1 — DESIGNING YOUR AGENTS
# ======================================================================

agent_one = Agent(
    role="CVFormatting",
    goal="Check whether this document is a usable CV, and if so, send it to the privacy agent. "
         "Default to accepting the document. Only reject it for one of these three specific reasons: "
         "(1) it is blank or contains only a few scattered words with no real content, "
         "(2) the extracted text is garbled, corrupted, or unreadable (broken encoding, repeated symbols, nonsensical fragments), or "
         "(3) it is clearly not a CV at all (e.g. a cover letter, an invoice, a random article, an empty template). "
         "Do not reject a CV for having an unconventional structure, missing section headers, unusual ordering, "
         "non-standard formatting, being short but still substantive, or covering only one or two typical areas "
         "(a CV that only lists skills and work history with no separate education section is still valid). "
         "A real CV can look very different from a template - it might be a single paragraph, a plain list of "
         "bullet points with no headers, or organized in a way you don't expect. As long as it is clearly about "
         "a real candidate's background, skills, or experience, treat it as valid regardless of how it's structured. "
         "Do not evaluate the quality, relevance, or strength of the candidate's experience - only whether the "
         "document is a real, readable CV usable by other agents. "
         "Do not make any changes to the document. Do not edit or summarise. ",

    backstory="You are an experienced software engineer who checks that uploaded documents are usable CVs before "
              "they move further down the pipeline. You've seen enough real resumes to know they come in every "
              "shape imaginable - single-paragraph summaries, unlabeled bullet lists, unconventional layouts, "
              "CVs translated awkwardly from another language, minimalist one-page formats - and you never reject "
              "a document just because it doesn't look like a template. "
              "You reserve rejection for the rare, clear-cut cases: the file is genuinely blank, the text came "
              "through corrupted or garbled, or the document plainly isn't a CV at all (a cover letter, an invoice, "
              "a random file uploaded by mistake). When in doubt, you let it through - a false rejection costs a "
              "real candidate their shot, while an odd-looking but valid CV downstream costs nothing. "
              "You're not here to judge quality or content, only to catch the documents that are truly unusable. ",
    llm=llm,
    verbose=VERBOSE,
    allow_delegation=False,
)

agent_two = Agent(
    role="PrivacyProtector",
    goal="You will receive a CV that has already been confirmed valid from the CVFormatting agent. "
         "You will need to remove all personally identifiable information from the CV and instead replace it with 'PII'. "
         "Separately, you will report the personal information you removed so it can be stored in a 'Contact Information' record for human auditors."
         "Applicant IDs will be assigned outside your tasks so you don't need to invent one. "
         "Personal information would include: Name, Age, Gender, Address, Email Address (any text with the @ symbol for example: sarah.whitfiel@email.com), Home Address, Phone Number(anything in the format of +44 7700 900123 or a string of numbers longer than 6 numbers), Date of Birth, Nationality, Religion, Marital Status, Sexual Orientation, any other demographic identifier, and any photos. "
         "Preserve the original formatting and structure as much as possible only substitute the information with PII. "
         "Do not make any other changes. Do not summarise. ",
    backstory="You are a compliance specialist who's seen how bias creeps into hiring the moment a name, photo, or address enters the picture. "
              "You believe candidates should be judged on skills alone, so you strip every CV of anything that could identify or bias evaluation: names, contact details, photos, addresses, ages, nationality, and personal social links. "
              "You're precise and conservative, when in doubt, you redact. "
              "But you never touch what matters: job titles, employers, education, skills, and dates. "
              "Your work is what makes fair, skills-based screening possible. "
              "You also ensure that all personal information is reported clearly so it can be stored elsewhere for human auditors to cross-check later. ",
    llm=llm,
    verbose=VERBOSE,
    allow_delegation=False,
)

agent_three = Agent(
    role="Senior CV Sentence Classification Specialist",

    goal=(
        "Read the full text of a single candidate CV and classify every "
        "substantive sentence or bullet point into exactly one of five "
        "categories — Education, Extra Curricular, Projects, Experiences, "
        "or Skills — preserving the candidate's original wording, without "
        "summarizing, judging, scoring, or inventing content, and return "
        "the result as strict JSON with all five categories present as keys."
    ),

    backstory=(
        "You spent 8 years as a technical recruiter and resume-screening "
        "consultant for Fortune 500 hiring teams, personally structuring "
        "over 20,000 CVs across engineering, product, and design roles. "
        "You are known for obsessive literal accuracy — you never "
        "paraphrase a candidate's own words, and you have a sharp instinct "
        "for telling a paid internship apart from a personal side project "
        "apart from a university club activity, even when the phrasing is "
        "ambiguous. You built your reputation by turning messy, "
        "inconsistently formatted resumes into perfectly organized category "
        "breakdowns that hiring managers could scan in seconds. You never "
        "drop a relevant line, and you never invent one that isn't in the "
        "source text."
    ),

    llm=llm,
    verbose=True,          # optional: prints the agent's reasoning steps
    allow_delegation=False # optional: this agent shouldn't hand work to others
)

agent_four = Agent(
    role="JobRequirementsAnalyst",
    goal=(
        "Read the full text of a job description and confirm it is actually a job desrciption and at the very least has some requirements listed of stated in it, if not reject the job description and deem it invalid. Once that is complete, read the full job description again and break it down into the same five categories used to classify candidate CVs - Education, Extra Curricular, Projects, Experiences, and Skills - so each category can be compared directly against the equivalent category in a classified CV. "
        "For each category, extract every specific, concrete requirement mentioned: required years of experience, specific tools, technologies, qualifications, degrees, certifications, or extracurricular expectations. "
        "Do not invent requirements that aren't stated, and do not drop requirements that are stated even if they're implied rather than explicit. "
        "If a category has no requirements mentioned, return it as an empty list rather than guessing at what might be expected. "
        "Return the result as strict JSON with all five categories present as keys."
    ),
    backstory=(
        "You've spent years reading job postings that bury the real requirements under vague "
        "corporate language, and you've learned to tell a genuine requirement ('5+ years of "
        "Python required') apart from a soft preference ('nice to have: familiarity with "
        "Python') without conflating the two. "
        "You know a job description is only useful for screening once its requirements are "
        "pulled out cleanly and organized the same way the candidates are - so you always sort "
        "what you find into the same five buckets used everywhere else in this pipeline: "
        "Education, Extra Curricular, Projects, Experiences, and Skills. "
        "You never pad a category with something you've inferred but that isn't actually "
        "written, and you never skip a requirement just because it's phrased unusually or "
        "buried in an odd part of the posting."
    ),
    llm=llm,
    verbose=VERBOSE,
    allow_delegation=False,
)

agent_five = Agent(
    role="ApplicantScorer",
    goal=(
        "Compare a candidate's classified CV against a job description's requirements, which "
        "have already been broken down into the same five categories: Education, Extra "
        "Curricular, Projects, Experiences, and Skills. "
        "For each category, assess how well the candidate's content meets the corresponding "
        "requirement, and assign a score from 0 to 100 with a one to two sentence "
        "justification that cites specific evidence from the CV. "
        "Then compute a single overall weighted score using these exact weights: Experiences "
        "30%, Skills 25%, Projects 20%, Education 15%, Extra Curricular 10%. "
        "Base every category score strictly on what's actually present in the CV and what's "
        "actually required - never invent qualifications the candidate hasn't demonstrated, "
        "never penalize a candidate for lacking something the job description never asked "
        "for, and never let strength in one category color your assessment of a different "
        "one. "
        "Return the result as strict JSON containing a score and justification for each of the "
        "five categories, plus the final weighted overall_score. Ensure consistency in scoring process i.e if you have seen a CV before and are scoring it again, make sure u give it the same score or at the very least within 5%"
    ),
    backstory=(
        "You've spent a decade on hiring panels and learned how easily one standout resume "
        "line can unfairly color a reviewer's read of everything else. "
        "You made your reputation by scoring category by category, deliberately, rather than "
        "forming one holistic gut impression - because that's exactly where bias creeps in. "
        "A weak Projects section never makes you harsher on someone's Experience, and a strong "
        "Skills section never makes you generous with their Education. "
        "You also respect that the weights you're given were chosen on purpose - Experience "
        "matters more than Extra Curricular for a reason - so you apply them exactly rather "
        "than letting your own instincts override them."
    ),
    llm=llm,
    verbose=VERBOSE,
    allow_delegation=False,
)

agent_six = Agent(
    role="CandidateRanker",
    goal=(
        "You will be given a list of candidates already sorted from best to worst fit by "
        "their overall_score, in that exact order - your job is not to re-order or re-score "
        "them. "
        "For each candidate, write a one-sentence rationale for their position, referencing "
        "their strongest and weakest scoring categories among Experiences, Skills, Projects, "
        "Education, and Extra Curricular. "
        "If two or more adjacent candidates have overall_score values within a few points of "
        "each other, say so explicitly rather than presenting a confident distinction the "
        "underlying scores don't actually support. "
        "Return the result as strict JSON: a list matching the given order, each entry "
        "containing applicant_id, rank, overall_score, and your rationale."
    ),
    backstory=(
        "You've sat on hiring panels where the hard part was never the math - it was "
        "explaining to everyone else in the room why the numbers landed where they did. "
        "By the time a list reaches you, the scoring and ordering are already settled, and "
        "second-guessing them isn't your job. "
        "What you're good at is turning a ranked list into something a panel can actually "
        "discuss: a specific, short reason for each candidate's position, and an honest flag "
        "whenever two candidates are close enough that the gap between them shouldn't be "
        "treated as decisive. "
        "You never manufacture more certainty than the scores in front of you actually "
        "support."
    ),
    llm=llm,
    verbose=VERBOSE,
    allow_delegation=False,
)

# ======================================================================
# STEP 2 — PUT YOUR AGENTS ON THE TEAM
# ----------------------------------------------------------------------
# List every agent you created so the manager knows who's available.
# ======================================================================

my_team = [
    agent_one,
    agent_two,
    agent_three,
    agent_four,
    agent_five,
    agent_six,
]

# ======================================================================
# STEP 3 — CREATING CLASSES FOR STRUCTURED OUTPUTS
# ----------------------------------------------------------------------
# Using pydantic BaseModel classes to help structure output so it is easy to read.
# Giving format-check and privacy check of the first two agents a fixed shape.
# This means code can branch on real fields instead of trying to read text files.
# ======================================================================

class FormatCheckResult(BaseModel):
    valid_format: bool  # Gives True/False on whether a CV is valid or not
    reason_invalid: str = ""    # Gives explanation on why a CV may be marked invalid

class PrivacyResult(BaseModel):
    redacted_cv: str    # The CV after all personal information has been redacted
    contact_information: str    # All personal information that has been removed from the CVs that can be used to contact an applicant


class CVClassification(BaseModel):
    education: list[str] = []
    extra_curricular: list[str] = []
    projects: list[str] = []
    experiences: list[str] = []
    skills: list[str] = []


class JobCriteria(BaseModel):
    education: list[str] = []
    extra_curricular: list[str] = []
    projects: list[str] = []
    experiences: list[str] = []
    skills: list[str] = []


class CategoryScore(BaseModel):
    score: float
    justification: str


class ApplicantScore(BaseModel):
    education: CategoryScore
    extra_curricular: CategoryScore
    projects: CategoryScore
    experiences: CategoryScore
    skills: CategoryScore
    overall_score: float


class RankedCandidate(BaseModel):
    applicant_id: str
    rank: int
    overall_score: float
    rationale: str


class RankingResult(BaseModel):
    rankings: list[RankedCandidate]

# ======================================================================
# STEP 4 — SPLITTING TASKS INTO STAGES
# ----------------------------------------------------------------------
# This is done to so that the agents can hand off objects to each other in order.
# ======================================================================

# -----------------------------------------------------------------------
# Stage 0: Checking Job Description
#   -> Breaking down job description into analysable components
# -----------------------------------------------------------------------

analyze_jd_task = Task(
    description=(
        "Here is the full text of a job description:\n\n"
        "{job_requirements}\n\n"
        "Break down its requirements into the same five categories used to classify candidate "
        "CVs: Education, Extra Curricular, Projects, Experiences, and Skills."
    ),
    expected_output=(
        "Strict JSON with all five categories present as keys, each containing a list of the "
        "specific requirements mentioned in that category (empty list if none are mentioned)."
    ),
    agent=agent_four,
    output_pydantic=JobCriteria,
)

jd_analysis_stage = Crew(
    agents=[agent_four],
    tasks=[analyze_jd_task],
    process=Process.sequential,
    verbose=VERBOSE,
    tracing=TRACING,
)

# -----------------------------------------------------------------------
# Stage 1: Format Check and Redaction
#   -> Only stage that sees original uploaded CV
#   -> Checks CV is valid, and if marked invalid gives a reason why it is
# -----------------------------------------------------------------------

format_task = Task(
    description=(
        "Here is the raw text extracted from an uploaded document: \n\n"
        "{original_cv} \n\n"
        "Decide whether this is a valid, usable CV: check that the text has been extracted cleanly. "
        "Check that the recognisable CV content (e.g. experience, education, skills) is present and there is enough content to evaluate."
    ),
    expected_output=(
        "A structured verdict: assigning valid_format to true/false, and a reason (if valid then empty string, and if invalid then a short explanation as to why)."
    ),
    agent=agent_one,
    output_pydantic=FormatCheckResult,
)

redact_task = Task(
    description=(
        "Here is the same document, that has been checked for validity in the previous step: \n\n"
        "{original_cv} \n\n"
        "Remove all personally identifiable information and replace it with 'PII'. "
        "Report all the personal information you removed separately, so it can be stored for human auditors."
    ),
    expected_output=(
        "A structured result: redacted_cv (full CV with personal information replaced with 'PII' and otherwise unchanged. Additionally, the contact_information (which contains all personal details you removed as a short readable list)."
    ),
    agent=agent_two,
    context=[format_task],
    output_pydantic=PrivacyResult,
)

privacy_stage = Crew(
    agents=[agent_one, agent_two],
    tasks=[format_task, redact_task],
    process=Process.sequential,
    verbose=VERBOSE,
    tracing=TRACING,
)

# -----------------------------------------------------------------------
# Stage 2: CV Matching
#   -> Breaks down CV into its main points
#   -> Matches CV to job application
#   -> Gives a score for each CV
# -----------------------------------------------------------------------

classify_task = Task(
    description=(
        "Here is a candidate's CV with all personal information already redacted: \n\n"
        "{redacted_cv} \n\n"
        "Classify every substantive sentence or bullet point into exactly one of: Education, Extra Curricular, Projects, Experiences, or Skills."
    ),
    expected_output=(
        "Strict JSON with all five categories present as keys, each containing a list of the candidate's original sentences/bullets classified into that category."
    ),
    agent=agent_three,
    output_pydantic=CVClassification,
)

score_task = Task(
    description=(
        "Here is a job description's requirements, already broken down into the same five "
        "categories used above:\n\n"
        "{job_criteria}\n\n"
        "Compare the candidate's classified CV from the previous step against these "
        "requirements, category by category, and score each one from 0 to 100 with a short "
        "justification. Then compute the overall_score using these exact weights: Experiences "
        "30%, Skills 25%, Projects 20%, Education 15%, Extra Curricular 10%."
    ),
    expected_output=(
        "Strict JSON with a score and justification for each of the five categories, plus the "
        "final weighted overall_score."
    ),
    agent=agent_five,
    context=[classify_task],
    output_pydantic=ApplicantScore,
)

screening_stage = Crew(
    agents=[agent_three, agent_five],
    tasks=[classify_task, score_task],
    process=Process.sequential,
    verbose=VERBOSE,
    tracing=TRACING,
)

# -----------------------------------------------------------------------
# Stage 2: CV Ranking
#   -> Ranks CV based on score
# -----------------------------------------------------------------------

rank_task = Task(
    description=(
        "Here is a list of candidates already sorted from best to worst fit by their "
        "overall_score - do not re-order or re-score them:\n\n"
        "{ranked_candidates}\n\n"
        "For each candidate, write a one-sentence rationale referencing their strongest and "
        "weakest scoring categories. If adjacent candidates have overall_score values within a "
        "few points of each other, note explicitly that the gap is narrow."
    ),
    expected_output=(
        "Strict JSON matching RankingResult: a 'rankings' list in the same given order, each "
        "entry containing applicant_id, rank, overall_score, and rationale."
    ),
    agent=agent_six,
    output_pydantic=RankingResult,
)

ranking_stage = Crew(
    agents=[agent_six],
    tasks=[rank_task],
    process=Process.sequential,
    verbose=VERBOSE,
    tracing=TRACING,
)


# ======================================================================
# STEP 4 — HELPERS + BATCH RUNNER
# ----------------------------------------------------------------------
# Each helper below does exactly one stage of the pipeline. run_batch just
# calls them in order, so it reads like a table of contents rather than
# doing any of the work itself.
# ======================================================================

def load_text(path):    # opens a file and returns its text
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def load_cvs(cv_paths):     # uses load_text on every file path in my list (cv_paths), skipping any files it fails to read (printing a warning). Creates two lists of filenames and raw_texts.
    filenames, raw_texts = [], []
    for path in cv_paths:
        try:
            raw_texts.append(load_text(path))
        except (UnicodeDecodeError, OSError) as e:
            print(f"Skipping {path}: could not read as text ({e})")
            continue
        filenames.append(path)
    return filenames, raw_texts


def analyze_job_requirements(job_requirements):     # Sends job description to agent four and splits it into criteria, once per loop.
    output = jd_analysis_stage.kickoff(inputs={"job_requirements": job_requirements})
    return output.pydantic if isinstance(output.pydantic, JobCriteria) else None


def run_privacy_stage(filenames, raw_texts):    # Stage 1: Once per CV: format check, then redact. Splits the results into contact_records (kept private) and valid_items (ready for screening).
    privacy_outputs = privacy_stage.kickoff_for_each(
        inputs=[{"original_cv": text} for text in raw_texts]
    )

    id_width = max(2, len(str(len(filenames))))  # "01"..."99", widening automatically past 99
    contact_records = {}
    valid_items = []  # (filename, applicant_id, redacted_cv_text)

    for i, (filename, crew_output) in enumerate(zip(filenames, privacy_outputs)):
        applicant_id = str(i + 1).zfill(id_width)

        format_result = next(
            (t.pydantic for t in crew_output.tasks_output if isinstance(t.pydantic, FormatCheckResult)),
            None,
        )
        redaction = next(
            (t.pydantic for t in crew_output.tasks_output if isinstance(t.pydantic, PrivacyResult)),
            None,
        )

        if redaction:
            contact_records[applicant_id] = redaction.contact_information

        if format_result and not format_result.valid_format:
            print(f"Rejected {filename} (applicant {applicant_id}): {format_result.reason_invalid}")
            continue

        if not redaction:
            print(f"Skipping {filename} (applicant {applicant_id}): redaction step returned no structured output")
            continue

        valid_items.append((filename, applicant_id, redaction.redacted_cv))

    return valid_items, contact_records


def save_contact_records(contact_records, contact_store_path):
    with open(contact_store_path, "w", encoding="utf-8") as f:
        json.dump(contact_records, f, indent=2)


def run_screening_stage(valid_items, job_criteria):     # Stage 2: Once per CV: classify into categories and then score against job criteria.
    job_criteria_json = job_criteria.model_dump_json() if job_criteria else "{}"

    screening_outputs = screening_stage.kickoff_for_each(
        inputs=[
            {"redacted_cv": cv_text, "job_criteria": job_criteria_json}
            for _, _, cv_text in valid_items
        ]
    )

    scored = []
    for (filename, applicant_id, _), crew_output in zip(valid_items, screening_outputs):
        score = next(
            (t.pydantic for t in crew_output.tasks_output if isinstance(t.pydantic, ApplicantScore)),
            None,
        )
        if not score:
            print(f"Skipping {filename} (applicant {applicant_id}): scoring step returned no structured output")
            continue

        scored.append({
            "source_file": filename,
            "applicant_id": applicant_id,
            "overall_score": score.overall_score,
            "category_scores": score.model_dump(),
        })

    return scored


def rank_candidates(scored_candidates): #Sort by overall score, then ask agent_six only to explain each position.
    ranked = sorted(scored_candidates, key=lambda c: c["overall_score"], reverse=True)
    for i, candidate in enumerate(ranked):
        candidate["rank"] = i + 1

    summary = [
        {"applicant_id": c["applicant_id"], "rank": c["rank"], "overall_score": c["overall_score"]}
        for c in ranked
    ]
    output = ranking_stage.kickoff(inputs={"ranked_candidates": json.dumps(summary)})
    result = output.pydantic if isinstance(output.pydantic, RankingResult) else None

    rationales = {r.applicant_id: r.rationale for r in result.rankings} if result else {}
    for candidate in ranked:
        candidate["rationale"] = rationales.get(candidate["applicant_id"], "")

    return ranked


def run_batch(job_description_path, cv_paths, contact_store_path="contact_information.json"):
    job_requirements = load_text(job_description_path)
    job_criteria = analyze_job_requirements(job_requirements)

    filenames, raw_texts = load_cvs(cv_paths)
    if not filenames:
        print("No readable CVs found in cv_paths")
        return []

    valid_items, contact_records = run_privacy_stage(filenames, raw_texts)
    save_contact_records(contact_records, contact_store_path)
    if not valid_items:
        print("No valid CVs made it past the format/privacy stage.")
        return []

    scored_candidates = run_screening_stage(valid_items, job_criteria)
    if not scored_candidates:
        print("No candidates made it through classification and scoring.")
        return []

    return rank_candidates(scored_candidates)


if __name__ == "__main__":
    JOB_DESCRIPTION_PATH = "/Users/rishi/Downloads/job-description.txt"  # path to job description file
    CV_PATHS = [
        "/Users/rishi/Downloads/cv_example1.txt",  # list every CV file path needed to be screened
        "/Users/rishi/Downloads/cv_example2.txt",
    ]

    screened = run_batch(JOB_DESCRIPTION_PATH, CV_PATHS)

    for r in screened:
        print(f"\n#{r['rank']} - applicant {r['applicant_id']} ({r['source_file']}) - {r['overall_score']}/100")
        print(r["rationale"])








# ======================================================================
# STEP 3 — HANDLE A REQUEST  (you usually DON'T need to change this)
# ----------------------------------------------------------------------
# The manager reads the request + history and delegates to the right agent.
# Notice the Task has NO `agent=` — that's on purpose. The manager picks.
# ======================================================================
#
# conversation = []   # remembers what's been said so far
#
# def handle(user_request):
#    history = "\n".join(conversation)
#    task = Task(
#        description=(
#            f"Conversation so far:\n{history if history else '(nothing yet)'}\n\n"
#            f"The user now says: '{user_request}'. Fulfil their request, "
#            "using the earlier conversation for context."
#        ),
#        expected_output="A clear, helpful answer for the user.",
#    )
#    crew = Crew(
#        agents=my_team,
#        tasks=[task],
#        process=Process.hierarchical,   # the manager/orchestrator
#        manager_llm=llm,
#        verbose=VERBOSE,
#        tracing=TRACING,
#    )
#    result = crew.kickoff()
#
#    conversation.append(f"User: {user_request}")
#    conversation.append(f"Assistant: {result}")
#    return result
#
#
# ======================================================================
# STEP 4 — CHAT LOOP  (customise the welcome text for your team!)
# ======================================================================
#
#if __name__ == "__main__":
#    # TODO: change these lines to describe YOUR team and give example prompts
#    print("My Agent Team is ready! Try things like:")
#    print("...")
#    print("...")
#    print("(type 'quit' to exit)\n")
#
#    while True:
#        request = input("Input: ").strip()
#        if request.lower() in {"quit", "exit"}:
#            print("Goodbye!")
#            break
#        if not request:
#            continue
#
#        result = handle(request)
#        print("\nFinal answer:\n" + str(result) + "\n")


