import re
from abc import ABC
from typing import Any
import json
from FL_prompt import *
from FL_tools import *
from repoSearcher.util.api_requests import num_tokens_from_messages
from repoSearcher.util.postprocess_data import extract_code_blocks, extract_locs_for_files, extract_func_locs_for_files
from repoSearcher.util.preprocess_data import (get_repo_files, get_full_file_paths_and_classes_and_functions, correct_file_paths,
                                      line_wrap_content, transfer_arb_locs_to_locs, show_project_structure,
                                      )



class FL(ABC):
    def __init__(self, instance_id, structure, problem_statement, **kwargs):
        self.structure = structure
        self.instance_id = instance_id
        self.problem_statement = problem_statement


class AFL(FL):
    def __init__(
            self,
            instance_id,
            structure,
            problem_statement,
            model_name,
            backend,
            logger,
            **kwargs,
    ):
        super().__init__(instance_id, structure, problem_statement)
        self.max_tokens = 4096
        self.model_name = model_name
        self.backend = backend
        self.logger = logger

        self.MAX_CONTEXT_LENGTH = 32768 #if "qwen2.5-7b" in model_name else 100000

    def _parse_top5_file(self, content: str) -> list[str]:
        if content.count("```") % 2 != 0:
            return []
        extracted_output = re.findall(r'```(?:.*?\n)?(.*?)```', str(content), re.DOTALL)
        if not extracted_output:
            return []
        lines = "\n".join(extracted_output).strip().split('\n')
        parsed_list = []
        for line in lines:
            if line.strip().endswith(".py"):
                parsed_list.append(line.strip())
        return parsed_list

    def _parse_output(self, content: str):
        extracted_output = re.search(r'```(?:.*?)\n(.*?)```', content, re.DOTALL).group(1)
        return extracted_output

    def _issue_clarify(self):
        from repoSearcher.util.model import make_model
        clarify_msg = bug_report_clarify_prompt.format(problem_statement=self.problem_statement)
        message = [
            {
                "role": "user",
                "content": clarify_msg
            }
        ]
        model = make_model(
            model=self.model_name,
            backend=self.backend,
            logger=self.logger,
            max_tokens=4096,
            temperature=0.85,
            batch_size=1,
        )
        traj = model.codegen(message, num_samples=1)[0]
        bug_report_clarified = self._parse_output(traj["response"])
        return bug_report_clarified

    def consturct_bug_file_list(self, file: list):
        bug_file_content = ""
        for name in file:
            class_content = get_classes_of_file(name, self.instance_id)
            class_list = eval(get_classes_of_file(name, self.instance_id))
            class_func_content = "[\n"
            for class_name in class_list:
                class_func = get_functions_of_class(class_name, self.instance_id)
                class_func_content += f"{class_name}: {class_func} \n"
            class_func_content += "]"
            file_func_content = get_functions_of_file(name, self.instance_id)
            single_file_context = f"file: {name} \n\t class: {class_content} \n\t static functions:  {file_func_content} \n\t class fucntions: {class_func_content}\n"
            bug_file_content += single_file_context
            # print(bug_file_content)
        return bug_file_content

    def localize(
            self, max_retry=10, file=None, mock=False
    ) -> tuple[list[str], Any, Any]:
        from repoSearcher.util.model import make_model
        max_try = max_retry
        bug_report = bug_report_template_wo_repo_struct.format(problem_statement=self.problem_statement).strip()
        system_msg = location_system_prompt.format(functions=location_tool_prompt, max_try=max_try)
        # construct first-order function call graph
        bug_file_content = self.consturct_bug_file_list(file)
        location_guidence_msg = location_guidence_prmpt.format(bug_file_list=bug_file_content,
                                                               pre_select_num=7,
                                                               top_n=5)
        # init sate and start search
        user_msg = f"""
                {bug_report}
                {location_guidence_msg}
                """

        message = [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg}
        ]

        model = make_model(
            model=self.model_name,
            backend=self.backend,
            logger=self.logger,
            max_tokens=self.max_tokens,
            temperature=0.0,
            batch_size=1,
        )
        traj = model.codegen(message, num_samples=1)[0]
        traj["prompt"] = message
        reason = traj["response"]
        current_tokens = traj["usage"]["completion_tokens"] + traj["usage"]["prompt_tokens"]

        message.append({
            "role": "assistant",
            "content": reason
        })
        message.append({
            "role": "user",
            "content": call_function_prompt + bug_file_content
        })

        location_summary_tokens = num_tokens_from_messages([{
            "role": "user",
            "content": location_summary.format(bug_file_list=bug_file_content)
        }], self.model_name) + self.max_tokens

        # search step
        for j in range(max_try):

            if current_tokens > self.MAX_CONTEXT_LENGTH - 3 * location_summary_tokens:
                message.pop()
                break
            try:
                traj = model.codegen(message, num_samples=1)[0]
                current_tokens += traj["usage"]["completion_tokens"] + traj["usage"]["prompt_tokens"]
            except Exception as e:
                if "Tokens" in str(e):  # Check if error message indicates context length issue
                    message.pop()
                    break

            content = traj["response"]
            print(content)
            message.append({
                "role": "assistant",
                "content": content
            })
            def filter_function_call(content):
                return content.replace("```python", "").replace("`", "").replace("'", "").replace('"', '')
            try:
                function_call = filter_function_call(content)
                function_name = function_call[:function_call.find('(')].strip()
                arguments = function_call[function_call.find('(') + 1:function_call.rfind(')')].strip()
                args = [arg.strip() for arg in re.split(r",\s*(?![^()]*\))", arguments)]

                # 确保参数数量不超过三个
                arg1, arg2, arg3 = (args + ['None'] * 3)[:3]
                if function_name == 'get_functions_of_class':
                    function_retval = get_functions_of_class(arg1, self.instance_id)
                elif function_name == 'get_code_of_class':
                    function_retval = get_code_of_class(arg1, arg2, self.instance_id)
                elif function_name == 'get_code_of_class_function':
                    function_retval = get_code_of_class_function(arg1, arg2, arg3, self.instance_id)
                elif function_name == 'get_code_of_file_function':
                    function_retval = get_code_of_file_function(arg1, arg2, self.instance_id)
                else:
                    break
                # print(function_retval)
                message.append({"role": "user", "content": function_retval})
                message.append({
                    "role": "user",
                    "content": call_function_prompt + bug_file_content
                })
            except Exception as e:
                print(e)
                message.append({
                    "role": "user",
                    "content": "Please call functions in the right format to get enough information for your final answer." + location_tool_prompt})

        # summary the locations
        message.append({
            "role": "user",
            "content": location_summary.format(bug_file_list=bug_file_content)
        })
        traj = model.codegen(message, num_samples=1)[0]
        traj["prompt"] = message
        raw_output = traj["response"]

        self.logger.info(raw_output)
        model_found_locs = extract_code_blocks(raw_output)
        model_found_locs_separated = extract_locs_for_files(
            model_found_locs, file
        )

        return (
            model_found_locs_separated,
            raw_output,
            traj,
        )

    def localize_line(
            self,
            file_names,
            func_locs,
            context_window: int = 10,
            add_space: bool = False,
            sticky_scroll: bool = False,
            no_line_number: bool = False,
            temperature: float = 0.0,
            num_samples: int = 1,
    ):
        from repoSearcher.util.api_requests import num_tokens_from_messages
        from repoSearcher.util.model import make_model
        self.max_tokens = 4096
        coarse_locs = func_locs
        # file_names = []
        if not isinstance(func_locs, dict):
            return [], {}, {}
        # for key, item in func_locs.items():
        #     file_names.append(key)

        file_contents = get_repo_files(self.structure, file_names)
        topn_content, file_loc_intervals = construct_topn_file_context(
            coarse_locs,
            file_names,
            file_contents,
            self.structure,
            context_window=context_window,
            loc_interval=True,
            add_space=add_space,
            sticky_scroll=sticky_scroll,
            no_line_number=no_line_number,
        )
        if no_line_number:
            template = obtain_relevant_code_combine_top_n_no_line_number_prompt
        else:
            template = obtain_relevant_code_combine_top_n_prompt
        message = template.format(
            problem_statement=self.problem_statement, file_contents=topn_content
        )
        self.logger.info(f"prompting with message:\n{message}")
        self.logger.info("=" * 80)
        assert num_tokens_from_messages(message, self.model_name) < self.MAX_CONTEXT_LENGTH

        model = make_model(
            model=self.model_name,
            backend=self.backend,
            logger=self.logger,
            max_tokens=self.max_tokens,
            temperature=temperature,
            batch_size=num_samples,
        )
        raw_trajs = model.codegen(message, num_samples=num_samples)

        # Merge trajectories
        raw_outputs = [raw_traj["response"] for raw_traj in raw_trajs]
        traj = {
            "prompt": message,
            "response": raw_outputs,
            "usage": {  # merge token usage
                "completion_tokens": sum(
                    raw_traj["usage"]["completion_tokens"] for raw_traj in raw_trajs
                ),
                "prompt_tokens": sum(
                    raw_traj["usage"]["prompt_tokens"] for raw_traj in raw_trajs
                ),
            },
        }
        model_found_locs_separated_in_samples = []
        for raw_output in raw_outputs:
            model_found_locs = extract_code_blocks(raw_output)
            model_found_locs_separated = extract_locs_for_files(
                model_found_locs, file_names
            )
            model_found_locs_separated_in_samples.append(model_found_locs_separated)

            self.logger.info(f"==== raw output ====")
            self.logger.info(raw_output)
            self.logger.info("=" * 80)
            print(raw_output)
            print("=" * 80)
            self.logger.info(f"==== extracted locs ====")
            for loc in model_found_locs_separated:
                self.logger.info(loc)
            self.logger.info("=" * 80)
        self.logger.info("==== Input coarse_locs")
        coarse_info = ""
        for fn, found_locs in coarse_locs.items():
            coarse_info += f"### {fn}\n"
            if isinstance(found_locs, str):
                coarse_info += found_locs + "\n"
            else:
                coarse_info += "\n".join(found_locs) + "\n"
        self.logger.info("\n" + coarse_info)
        if len(model_found_locs_separated_in_samples) == 1:
            model_found_locs_separated_in_samples = (
                model_found_locs_separated_in_samples[0]
            )

        return (
            model_found_locs_separated_in_samples,
            {"raw_output_loc": raw_outputs},
            traj,
        )

    def file_localize_without_collect(
            self, max_retry=10, mock=False
    ) -> tuple[list[str], Any, Any]:
        # lazy import, not sure if this is actually better?

        from repoSearcher.util.model import make_model
        max_try = max_retry
        all_files = get_all_of_files(self.instance_id)
        # clarified_issue = self._issue_clarify()

        bug_report = bug_report_template.format(problem_statement=self.problem_statement,
                                                structure=all_files)
        # bug_report = bug_report_template.format(problem_statement=clarified_issue,
        #                                         structure=all_files)

        system_msg = file_system_prompt_without_tool
        guidence_msg = file_guidence_prmpt_without_tool.format(pre_select_num=int(max_try * 0.75),
                                                               top_n=int(max_try / 2))
        user_msg = f"""
                {bug_report}
                {guidence_msg}
                """
        message = [
            {
                "role": "system",
                "content": system_msg
            },
            {
                "role": "user",
                "content": user_msg
            }
        ]
        model = make_model(
            model=self.model_name,
            backend=self.backend,
            logger=self.logger,
            max_tokens=self.max_tokens,
            temperature=0.85,
            batch_size=1,
        )
        traj = model.codegen(message, num_samples=1)[0]
        traj["prompt"] = message
        raw_output = traj["response"]

        reason = raw_output
        message.append({
            "role": "assistant",
            "content": reason
        })
        message.append({
            "role": "user",
            "content": file_summary
        })
        traj = model.codegen(message, num_samples=1)[0]
        traj["prompt"] = message
        raw_output = traj["response"]

        self.logger.info(raw_output)
        model_found_files = self._parse_top5_file(raw_output)
        files, classes, functions = get_full_file_paths_and_classes_and_functions(
            self.structure
        )
        found_files = correct_file_paths(model_found_files, files)

        return (
            found_files,
            raw_output,
            traj,
        )

    def file_localize(self, max_retry=10, mock=False):
        from repoSearcher.util.api_requests import num_tokens_from_messages
        from repoSearcher.util.model import make_model
        all_files = get_all_of_files(self.instance_id)
        # bug_report = bug_report_template.format(problem_statement=self.problem_statement,
        #                                         structure=all_files.strip())
        bug_report = bug_report_template.format(problem_statement=self.problem_statement,
                                                structure=show_project_structure(self.structure).strip())

        system_msg = file_system_prompt_without_tool
        guidence_msg = file_guidence_prmpt_without_tool.format(pre_select_num=int(max_retry * 0.75),
                                                               top_n=int(max_retry / 2))
        user_msg = f"""
{bug_report}
{guidence_msg}
{file_summary}
"""

        message = [
            {
                "role": "system",
                "content": system_msg
            },
            {
                "role": "user",
                "content": user_msg
            }
        ]
        self.logger.info(f"prompting with message:\n{message}")
        self.logger.info("=" * 80)
        if mock:
            self.logger.info("Skipping querying model since mock=True")
            traj = {
                "prompt": message,
                "usage": {
                    "prompt_tokens": num_tokens_from_messages(message, self.model_name),
                },
            }
            return [], {"raw_output_loc": ""}, traj

        model = make_model(
            model=self.model_name,
            backend=self.backend,
            logger=self.logger,
            max_tokens=self.max_tokens,
            temperature=0,
            batch_size=1,
        )
        try:
            traj = model.codegen(message, num_samples=1)[0]
            traj["prompt"] = message
            raw_output = traj["response"]
        except:
            bug_report = bug_report_template.format(problem_statement=self.problem_statement,
                                                    structure=show_project_structure(self.structure).strip())
            user_msg = f"""
{bug_report}
{guidence_msg}
{file_summary}
"""
            message = [{"role": "system", "content": system_msg},
                       {"role": "user", "content": user_msg}]
            traj = model.codegen(message, num_samples=1)[0]
            traj["prompt"] = message
            raw_output = traj["response"]
        self.logger.info(raw_output)
        model_found_files = self._parse_top5_file(raw_output)

        import difflib
        def get_best_match(file: str, all_files: list[str], cutoff: float = 0.8) -> str:
            if file in all_files:
                return file
            matches = difflib.get_close_matches(file, all_files, n=1, cutoff=cutoff)
            return matches[0] if matches else file

        found_files = [f for f in model_found_files if f in all_files]

        if len(found_files) == 0:
            corrcted_tpl = format_correct_prompt.format(res=raw_output)
            formated_res = model.codegen([{"role": "user", "content": corrcted_tpl}], num_samples=1)[0]["response"]
            self.logger.info(formated_res)
            model_found_files = self._parse_top5_file(formated_res)
            found_files = [f for f in model_found_files if f in all_files]
            if len(found_files) == 0:
                found_files = [get_best_match(f, all_files) for f in model_found_files]

        reflection_result = model.codegen([{"role": "user", "content": file_reflection_prompt.format(problem_statement=self.problem_statement,
                                                    structure=show_project_structure(self.structure).strip(), pre_files=found_files)}],
                                          num_samples=1)[0]["response"]
        self.logger.info(reflection_result)
        reflection_files = self._parse_top5_file(reflection_result)
        reflection_files = [f for f in reflection_files if f in all_files]
        if len(reflection_files) == 0:
            reflection_files = [get_best_match(f, all_files) for f in reflection_files]


        return (
            reflection_files,
            {"raw_output_files": raw_output},
            traj,
        )

    def localize_with_p(
            self, max_retry=10, file=None, mock=False
    ) -> tuple[list[str], Any, Any]:
        func_loc_trajs = []
        from repoSearcher.util.model import make_model
        max_try = max_retry
        bug_report = bug_report_template_wo_repo_struct.format(problem_statement=self.problem_statement).strip()
        system_msg = location_system_prompt.format(functions=location_tool_prompt, max_try=max_try)
        # construct first-order function call graph
        bug_file_content = self.consturct_bug_file_list(file)
        location_guidence_msg = location_guidence_prmpt.format(bug_file_list=bug_file_content,
                                                               pre_select_num=7,
                                                               top_n=5)
        # init search state
        user_msg = f"""
                {bug_report}
                {location_guidence_msg}
                """

        message = [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg}
        ]

        model = make_model(
            model=self.model_name,
            backend=self.backend,
            logger=self.logger,
            max_tokens=self.max_tokens,
            temperature=0.0,
            batch_size=1,
        )
        traj = model.codegen(message, num_samples=1)[0]
        traj["prompt"] = message.copy()
        func_loc_trajs.append(traj)
        reason = traj["response"]
        current_tokens = traj["usage"]["completion_tokens"] + traj["usage"]["prompt_tokens"]

        message.append({
            "role": "assistant",
            "content": reason
        })
        message.append({
            "role": "user",
            "content": call_function_prompt + bug_file_content
        })

        location_summary_tokens = num_tokens_from_messages([{
            "role": "user",
            "content": location_summary.format(bug_file_list=bug_file_content)
        }], self.model_name) + self.max_tokens

        # Seach step
        for j in range(max_try):

            if current_tokens > self.MAX_CONTEXT_LENGTH - 3 * location_summary_tokens:
                message.pop()
                break
            try:
                traj = model.codegen(message, num_samples=1)[0]
                traj['prompt'] = message.copy()
                func_loc_trajs.append(traj)
                current_tokens += traj["usage"]["completion_tokens"] + traj["usage"]["prompt_tokens"]
            except Exception as e:
                if "Tokens" in str(e):  # Check if error message indicates context length issue
                    message.pop()
                    break

            content = traj["response"]
            print(content)
            message.append({
                "role": "assistant",
                "content": content
            })
            def filter_function_call(content):
                return content.replace("```python", "").replace("`", "").replace("'", "").replace('"', '')
            try:
                function_call = filter_function_call(content)
                function_name = function_call[:function_call.find('(')].strip()
                arguments = function_call[function_call.find('(') + 1:function_call.rfind(')')].strip()
                args = [arg.strip() for arg in re.split(r",\s*(?![^()]*\))", arguments)]

                # 确保参数数量不超过三个
                arg1, arg2, arg3 = (args + ['None'] * 3)[:3]
                if function_name == 'get_functions_of_class':
                    function_retval = get_functions_of_class(arg1, self.instance_id)
                elif function_name == 'get_code_of_class':
                    function_retval = get_code_of_class(arg1, arg2, self.instance_id)
                elif function_name == 'get_code_of_class_function':
                    function_retval = get_code_of_class_function(arg1, arg2, arg3, self.instance_id)
                elif function_name == 'get_code_of_file_function':
                    function_retval = get_code_of_file_function(arg1, arg2, self.instance_id)
                else:
                    break

                # pruner agent
                check_func_retval_prompt = f"""
You will be presented with a bug report with repository structure to access the source code of the system under test (SUT).
Your task is to locate the most likely culprit functions/classes based on the bug report.
<bug report>
{self.problem_statement}
</bug report>

Here is a result of a function/class code retrived by '{content}'.
Please check if the code is related to the bug and if the code should be added into context.
<code>
{function_retval}
</code>
Return True if the code is related to the bug and should be added into context, otherwise return False.
Since your answer will be processed automatically, please give your answer in the format as follows.
The returned content should be wrapped with ```.
```
True
```
or
```
False
```
"""
                traj = model.codegen([{"role": "user", "content": check_func_retval_prompt}], num_samples=1)[0]
                traj['prompt'] = [{"role": "user", "content": check_func_retval_prompt}]
                func_loc_trajs.append(traj)
                check_res = traj['response']
                flag = self._parse_output(check_res).strip()
                print(flag)
                if flag == "True":
                    message.append({"role": "user", "content": function_retval})
                    # search the next function node
                    message.append({
                        "role": "user",
                        "content": call_function_prompt + "\nYou can check the function it calls.\n" + bug_file_content
                    })
                else:
                    # pruning the context
                    message.append({"role": "user",
                                    "content": "I have already checked this function/class is not related to the bug. Don't check the functions it calls."})
                    message.append({"role": "user", "content": function_retval})
                    message.append({
                        "role": "user",
                        "content": call_function_prompt
                    })

            except Exception as e:
                print(e)
                message.append({
                    "role": "user",
                    "content": "Please call functions in the right format to get enough information for your final answer." + location_tool_prompt})

        # summary the locs
        message.append({
            "role": "user",
            "content": location_summary.format(bug_file_list=bug_file_content)
        })
        traj = model.codegen(message, num_samples=1)[0]
        traj["prompt"] = message.copy()
        func_loc_trajs.append(traj)
        raw_output = traj["response"]

        self.logger.info(raw_output)
        model_found_locs = extract_code_blocks(raw_output)
        model_found_locs_separated = extract_locs_for_files(
            model_found_locs, file
        )

        return (
            model_found_locs_separated,
            raw_output,
            traj,
            func_loc_trajs,
        )

    def construct_func_loc_prompt(
            self, max_retry=10, file=None, mock=False
    ) -> tuple[list[str], Any, Any]:
        func_loc_trajs = []
        from repoSearcher.util.model import make_model
        max_try = max_retry
        bug_report = bug_report_template_wo_repo_struct.format(problem_statement=self.problem_statement).strip()
        system_msg = location_system_prompt.format(functions=location_tool_prompt, max_try=max_try)
        # construct first-order function call graph
        bug_file_content = self.consturct_bug_file_list(file)
        location_guidence_msg = location_guidence_prmpt_cot_and_func_call.format(bug_file_list=bug_file_content,
                                                                                top_n=5)
                # init search state
        user_msg = f"""
                {bug_report}
                {location_guidence_msg}
                """

        message = [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg}
        ]

        return message
    
    def localize_with_only_function_call(
            self, max_retry=10, file=None, mock=False
    ) -> tuple[list[str], Any, Any]:
        func_loc_trajs = []
        from repoSearcher.util.model import make_model
        max_try = max_retry
        bug_report = bug_report_template_wo_repo_struct.format(problem_statement=self.problem_statement).strip()
        system_msg = location_system_prompt.format(functions=location_tool_prompt, max_try=max_try)
        # construct first-order function call graph
        bug_file_content = self.consturct_bug_file_list(file)
        location_guidence_msg = location_guidence_prmpt_cot_and_func_call.format(bug_file_list=bug_file_content,
                                                                                top_n=5)
                # init search state
        user_msg = f"""
                {bug_report}
                {location_guidence_msg}
                """

        message = [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg}
        ]

        model = make_model(
            model=self.model_name,
            backend=self.backend,
            logger=self.logger,
            max_tokens=self.max_tokens,
            temperature=0.0,
            batch_size=1,
        )
        # traj = model.codegen(message, num_samples=1)[0]
        # traj["prompt"] = message.copy()
        # func_loc_trajs.append(traj)
        # reason = traj["response"]
        # current_tokens = traj["usage"]["completion_tokens"] + traj["usage"]["prompt_tokens"]

        # message.append({
        #     "role": "assistant",
        #     "content": reason
        # })
        # message.append({
        #     "role": "user",
        #     "content": call_function_prompt + bug_file_content
        # })

        location_summary_tokens = num_tokens_from_messages([{
            "role": "user",
            "content": location_summary.format(bug_file_list=bug_file_content)
        }], self.model_name) + self.max_tokens

        current_tokens = num_tokens_from_messages(message, self.model_name)

        finish_flag = False
        # Seach step
        for j in range(max_try):
            if finish_flag:
                break
            # print(current_tokens)
            # breakpoint()
            if current_tokens > self.MAX_CONTEXT_LENGTH - 3 * location_summary_tokens:
                message.pop()
                break
            try:
                traj = model.codegen(message, num_samples=1)[0]
                traj['prompt'] = message.copy()
                func_loc_trajs.append(traj)
                # current_tokens += traj["usage"]["completion_tokens"] + traj["usage"]["prompt_tokens"]
            except Exception as e:
                if "Tokens" in str(e):  # Check if error message indicates context length issue
                    message.pop()
                    break

            content = traj["response"]
            # only get the first action
            if "</action>" in content:
                content = content.split("</action>")[0] + "</action>"
            message.append({
                "role": "assistant",
                "content": content
            })
            current_tokens = num_tokens_from_messages(message, self.model_name)
            def filter_function_call(content):
                import re
                matches = re.findall(r'<action>\n```(.*?)```\n</action>', content, re.DOTALL)
                if matches:
                    return matches[0]
                else:
                    return content
                # return content.replace("```python", "").replace("`", "").replace("'", "").replace('"', '')
            try:
                function_call = filter_function_call(content)
                function_name = function_call[:function_call.find('(')].strip()
                arguments = function_call[function_call.find('(') + 1:function_call.rfind(')')].strip()
                args = [arg.strip() for arg in re.split(r",\s*(?![^()]*\))", arguments)]
                # breakpoint()
                # 确保参数数量不超过三个
                arg1, arg2, arg3 = (args + ['None'] * 3)[:3]
                if function_name == 'get_functions_of_class':
                    arg1 = arg1.replace("'", "").replace('"', "")
                    function_retval = get_functions_of_class(arg1, self.instance_id)
                elif function_name == 'get_code_of_class':
                    arg1 = arg1.replace("'", "").replace('"', "")
                    arg2 = arg2.replace("'", "").replace('"', "")
                    function_retval = get_code_of_class(arg1, arg2, self.instance_id)
                elif function_name == 'get_code_of_class_function':
                    arg1 = arg1.replace("'", "").replace('"', "")
                    arg2 = arg2.replace("'", "").replace('"', "")
                    arg3 = arg3.replace("'", "").replace('"', "")
                    function_retval = get_code_of_class_function(arg1, arg2, arg3, self.instance_id)
                elif function_name == 'get_code_of_file_function':
                    arg1 = arg1.replace("'", "").replace('"', "")
                    arg2 = arg2.replace("'", "").replace('"', "")
                    function_retval = get_code_of_file_function(arg1, arg2, self.instance_id)
                elif function_name == 'exit':
                    finish_flag = True
                    break
                else:
                    break
                
                # breakpoint()
                check_func_prompt = f"""
Here is a result of a function/class code retrived by '{function_call}'.
<code>
{function_retval}
</code>
Your goal is to locate the culprit locations for the bug. If you need more information, you can call previous functions.
If you are confident of the answer, you can call the ```exit()``` function to finish the information seeking.
"""
                message.append({
                    "role": "user",
                    "content": check_func_prompt
                })
                current_tokens = num_tokens_from_messages(message, self.model_name)

            except Exception as e:
                print(e)
                message.append({
                    "role": "user",
                    "content": "Please call functions in the right format to get enough information for your final answer." + location_tool_prompt})
                current_tokens = num_tokens_from_messages(message, self.model_name)


        while current_tokens > self.MAX_CONTEXT_LENGTH - self.max_tokens - location_summary_tokens and message:
            message.pop()

        # summary the locs
        message.append({
            "role": "user",
            "content": location_summary.format(bug_file_list=bug_file_content)
        })
        traj = model.codegen(message, num_samples=1)[0]
        traj["prompt"] = message.copy()
        func_loc_trajs.append(traj)
        raw_output = traj["response"]

        self.logger.info(raw_output)
        model_found_locs = extract_code_blocks(raw_output)
        model_found_locs_separated = extract_locs_for_files(
            model_found_locs, file
        )

        return (
            model_found_locs_separated,
            raw_output,
            traj,
            func_loc_trajs,
        )

    def localize_with_native_function_call(
            self, max_retry=10, file=None, mock=False
    ) -> tuple[list[str], Any, Any]:
        func_loc_trajs = []
        from repoSearcher.util.model import make_model
        max_try = max_retry
        bug_report = bug_report_template_wo_repo_struct.format(problem_statement=self.problem_statement).strip()
        system_msg = location_system_prompt_native_function_call
        # construct first-order function call graph
        bug_file_content = self.consturct_bug_file_list(file)
        location_guidence_msg = location_guidence_prompt_native_function_call.format(bug_file_list=bug_file_content)
                # init search state
        user_msg = f"""
                {bug_report}
                {location_guidence_msg}
                """

        message = [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg}
        ]

        model = make_model(
            model=self.model_name,
            backend=self.backend,
            logger=self.logger,
            max_tokens=self.max_tokens,
            temperature=0.0,
            batch_size=1,
        )
        # traj = model.codegen(message, num_samples=1)[0]
        # traj["prompt"] = message.copy()
        # func_loc_trajs.append(traj)
        # reason = traj["response"]
        # current_tokens = traj["usage"]["completion_tokens"] + traj["usage"]["prompt_tokens"]

        # message.append({
        #     "role": "assistant",
        #     "content": reason
        # })
        # message.append({
        #     "role": "user",
        #     "content": call_function_prompt + bug_file_content
        # })

        location_summary_tokens = num_tokens_from_messages([{
            "role": "user",
            "content": location_summary.format(bug_file_list=bug_file_content)
        }], self.model_name) + self.max_tokens

        current_tokens = num_tokens_from_messages(message, self.model_name)

        finish_flag = False
        # Seach step
        for j in range(max_try):
            if finish_flag:
                break
            # print(current_tokens)
            # breakpoint()
            if current_tokens > self.MAX_CONTEXT_LENGTH - 3 * location_summary_tokens:
                message.pop()
                break
            try:
                traj = model.codegen(message, num_samples=1)[0]
                traj['prompt'] = message.copy()
                func_loc_trajs.append(traj)
                # current_tokens += traj["usage"]["completion_tokens"] + traj["usage"]["prompt_tokens"]
            except Exception as e:
                print(e)
                if "Tokens" in str(e):  # Check if error message indicates context length issue
                    message.pop()
                    break
            content = traj["response"]
            # only get the first action
            if "</tool_call>" in content:
                content = content.split("</tool_call>")[0] + "</tool_call>"
            message.append({
                "role": "assistant",
                "content": content
            })
            current_tokens = num_tokens_from_messages(message, self.model_name)
            
            try:
                from sglang.srt.function_call.function_call_parser import FunctionCallParser
            except ImportError:
                from sglang.srt.function_call_parser import FunctionCallParser
            from sglang.srt.openai_api.protocol import Tool
            from json import JSONDecodeError

            tool_schemas = [{'type': 'function', 'function': {'name': 'get_code_of_class', 'description': 'A tool for geting the class content of specific file', 'parameters': {'type': 'object', 'properties': {'file_name': {'type': 'string', 'description': 'The full file name', 'enum': None}, 'class_name': {'type': 'string', 'description': 'class name', 'enum': None}}, 'required': ['file_name', 'class_name']}, 'strict': True}}, {'type': 'function', 'function': {'name': 'get_code_of_class_function', 'description': 'A tool for geting the function content of specific class', 'parameters': {'type': 'object', 'properties': {'file_name': {'type': 'string', 'description': 'The full file name', 'enum': None}, 'class_name': {'type': 'string', 'description': 'class name', 'enum': None}, 'func_name': {'type': 'string', 'description': 'function name', 'enum': None}}, 'required': ['file_name', 'class_name', 'func_name']}, 'strict': True}}, {'type': 'function', 'function': {'name': 'get_code_of_file_function', 'description': 'A tool for geting the function content of specific file', 'parameters': {'type': 'object', 'properties': {'file_name': {'type': 'string', 'description': 'The full file name', 'enum': None}, 'func_name': {'type': 'string', 'description': 'function name', 'enum': None}}, 'required': ['file_name', 'func_name']}, 'strict': True}}, {'type': 'function', 'function': {'name': 'exit', 'description': 'exit if you have found all the information needed.', 'parameters': {}}}]

            tool_call_parser_type = "qwen25"
            sgl_tools = [Tool.model_validate(tool_schema) for tool_schema in tool_schemas]
            function_call_parser = FunctionCallParser(
                sgl_tools,
                tool_call_parser_type,
            )
            if "exit()" in content:
                finish_flag = True
                break
            if function_call_parser.has_tool_call(content):
                try:
                    normed_content, tool_calls = function_call_parser.parse_non_stream(content)
                except JSONDecodeError:
                    normed_content = content
                    tool_calls = []
                except AttributeError:
                    normed_content = content
                    tool_calls = []
                try:
                    tool_call = tool_calls[0]
                    tool_name = tool_call.name
                    tool_args = json.loads(tool_call.parameters)
                    if tool_name == 'get_code_of_class':
                        tool_args['file_name'] = tool_args['file_name'].replace("'", "").replace('"', "")
                        tool_args['class_name'] = tool_args['class_name'].replace("'", "").replace('"', "")
                        function_call = f"{tool_name}({tool_args['file_name']}, {tool_args['class_name']})"
                        function_retval = get_code_of_class(tool_args['file_name'], tool_args['class_name'], self.instance_id)
                    elif tool_name == 'get_code_of_class_function':
                        tool_args['file_name'] = tool_args['file_name'].replace("'", "").replace('"', "")
                        tool_args['class_name'] = tool_args['class_name'].replace("'", "").replace('"', "")
                        tool_args['func_name'] = tool_args['func_name'].replace("'", "").replace('"', "")
                        function_call = f"{tool_name}({tool_args['file_name']}, {tool_args['class_name']}, {tool_args['func_name']})"
                        function_retval = get_code_of_class_function(tool_args['file_name'], tool_args['class_name'], tool_args['func_name'], self.instance_id)
                    elif tool_name == 'get_code_of_file_function':
                        tool_args['file_name'] = tool_args['file_name'].replace("'", "").replace('"', "")
                        tool_args['func_name'] = tool_args['func_name'].replace("'", "").replace('"', "")
                        function_call = f"{tool_name}({tool_args['file_name']}, {tool_args['func_name']})"
                        function_retval = get_code_of_file_function(tool_args['file_name'], tool_args['func_name'], self.instance_id)
                    elif tool_name == 'exit':
                        finish_flag = True
                        break
                    else:
                        raise ValueError(f"Unknown tool name: {tool_name}")
                
                    check_func_prompt = f"""
Here is a result of a function/class code retrived by '{function_call}'.
<code>
{function_retval}
</code>
Your goal is to locate the culprit locations for the bug. If you need more information, you can call previous functions.
If you are confident of the answer, you can call the ```exit()``` function to finish the information seeking.
"""
                    message.append({
                        "role": "user",
                        "content": check_func_prompt
                    })
                    current_tokens = num_tokens_from_messages(message, self.model_name)
                except Exception as e:
                    print(e)
                    message.append({
                    "role": "user",
                    "content": """wrong tool call format! For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:
For example:
<tool_call>
{"name": "get_code_of_class", "arguments": {    "file_name": "file1.py",    "class_name": "class1"  }}
</tool_call>
Note: all the arguments must be filled in."""})
                current_tokens = num_tokens_from_messages(message, self.model_name)
            else:
                message.append({
                    "role": "user",
                    "content": """wrong tool call format! For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:
For example:
<tool_call>
{"name": "get_code_of_class", "arguments": {    "file_name": "file1.py",    "class_name": "class1"  }}
</tool_call>
Note: all the arguments must be filled in."""})
                current_tokens = num_tokens_from_messages(message, self.model_name)


        while current_tokens > self.MAX_CONTEXT_LENGTH - self.max_tokens - location_summary_tokens and message:
            message.pop()

        # summary the locs
        message.append({
            "role": "user",
            "content": location_summary.format(bug_file_list=bug_file_content)
        })
        traj = model.codegen(message, num_samples=1)[0]
        traj["prompt"] = message.copy()
        func_loc_trajs.append(traj)
        raw_output = traj["response"]

        self.logger.info(raw_output)
        model_found_locs = extract_code_blocks(raw_output)
        model_found_locs_separated = extract_locs_for_files(
            model_found_locs, file
        )

        return (
            model_found_locs_separated,
            raw_output,
            traj,
            func_loc_trajs,
        )

    def file_localize_with_cosil(self, max_retry=10, mock=False):
        file_loc_trajs = []
        from repoSearcher.util.api_requests import num_tokens_from_messages
        from repoSearcher.util.model import make_model
        all_files = get_all_of_files(self.instance_id)
        bug_report = bug_report_template.format(problem_statement=self.problem_statement,
                                                structure=show_project_structure(self.structure).strip())

        system_msg = file_system_prompt_without_tool
        guidence_msg = file_guidence_prmpt_without_tool_cosil.format(pre_select_num=int(max_retry * 0.75),
                                                               top_n=int(max_retry / 2))
        user_msg = f"""
{bug_report}
{guidence_msg}
{file_summary}
"""

        message = [
            {
                "role": "system",
                "content": system_msg
            },
            {
                "role": "user",
                "content": user_msg
            }
        ]
        self.logger.info(f"prompting with message:\n{message}")
        self.logger.info("=" * 80)
        if mock:
            self.logger.info("Skipping querying model since mock=True")
            traj = {
                "prompt": message,
                "usage": {
                    "prompt_tokens": num_tokens_from_messages(message, self.model_name),
                },
            }
            file_loc_trajs = [traj]
            return [], {"raw_output_loc": ""}, traj, file_loc_trajs

        model = make_model(
            model=self.model_name,
            backend=self.backend,
            logger=self.logger,
            max_tokens=self.max_tokens,
            temperature=0,
            batch_size=1,
        )

        traj = model.codegen(message, num_samples=1)[0]
        traj["prompt"] = message
        raw_output = traj["response"]
        file_loc_trajs.append(traj)

        self.logger.info(raw_output)
        model_found_files = self._parse_top5_file(raw_output)

        import difflib
        def get_best_match(file: str, all_files: list[str], cutoff: float = 0.8) -> str:
            if file in all_files:
                return file
            matches = difflib.get_close_matches(file, all_files, n=1, cutoff=cutoff)
            return matches[0] if matches else file

        found_files = [f for f in model_found_files if f in all_files]

        # reflection to correct format
        if len(found_files) == 0:
            corrcted_tpl = format_correct_prompt.format(res=raw_output)
            traj = model.codegen([{"role": "user", "content": corrcted_tpl}], num_samples=1)[0]
            formated_res = ["response"]
            traj['prompt'] = [{"role": "user", "content": corrcted_tpl}]
            file_loc_trajs.append(traj)
            self.logger.info(formated_res)
            model_found_files = self._parse_top5_file(formated_res)
            found_files = [f for f in model_found_files if f in all_files]
            if len(found_files) == 0:
                found_files = [get_best_match(f, all_files) for f in model_found_files]

        # extract the first-order module graph context
        import_content = ""
        _parsed_path = []
        for loc in found_files:
            if loc in _parsed_path:
                continue
            import_content += f"file: {loc}\n {get_imports_of_file(loc, self.instance_id)}\n"
            _parsed_path.append(loc)

        # reflection with module call graph
        reflection_message = [{"role": "user", "content": file_reflection_prompt_cosil.format(problem_statement=self.problem_statement,
                                                                       structure=show_project_structure(
                                                                           self.structure).strip(),
                                                                       import_content=import_content,
                                                                       pre_files=found_files)}]
        traj = model.codegen(
            reflection_message,
            num_samples=1)[0]
        traj["prompt"] = reflection_message
        file_loc_trajs.append(traj)
        reflection_result = traj["response"]
        self.logger.info(reflection_result)
        reflection_files = self._parse_top5_file(reflection_result)
        reflection_files = [f for f in reflection_files if f in all_files]
        if len(reflection_files) == 0:
            reflection_files = [get_best_match(f, all_files) for f in reflection_files]

        return (
            reflection_files,
            {"raw_output_files": raw_output},
            traj,
            file_loc_trajs,
        )

    def file_localize_with_g(self, max_retry=10, mock=False,use_cosil=False):
        file_loc_trajs = []
        from repoSearcher.util.api_requests import num_tokens_from_messages
        from repoSearcher.util.model import make_model
        all_files = get_all_of_files(self.instance_id)
        bug_report = bug_report_template.format(problem_statement=self.problem_statement,
                                                structure=show_project_structure(self.structure).strip())

        system_msg = file_system_prompt_without_tool
        guidence_msg = file_guidence_prmpt_without_tool.format(pre_select_num=int(max_retry * 0.75),
                                                               top_n=int(max_retry / 2))
        user_msg = f"""
{bug_report}
{guidence_msg}
{file_summary}
"""

        message = [
            {
                "role": "system",
                "content": system_msg
            },
            {
                "role": "user",
                "content": user_msg
            }
        ]
        self.logger.info(f"prompting with message:\n{message}")
        self.logger.info("=" * 80)
        if mock:
            self.logger.info("Skipping querying model since mock=True")
            traj = {
                "prompt": message,
                "usage": {
                    "prompt_tokens": num_tokens_from_messages(message, self.model_name),
                },
            }
            file_loc_trajs = [traj]
            return [], {"raw_output_loc": ""}, traj, file_loc_trajs

        model = make_model(
            model=self.model_name,
            backend=self.backend,
            logger=self.logger,
            max_tokens=self.max_tokens,
            temperature=0,
            batch_size=1,
        )

        traj = model.codegen(message, num_samples=1)[0]
        traj["prompt"] = message
        raw_output = traj["response"]
        file_loc_trajs.append(traj)

        self.logger.info(raw_output)
        model_found_files = self._parse_top5_file(raw_output)

        import difflib
        def get_best_match(file: str, all_files: list[str], cutoff: float = 0.8) -> str:
            if file in all_files:
                return file
            matches = difflib.get_close_matches(file, all_files, n=1, cutoff=cutoff)
            return matches[0] if matches else file

        found_files = [f for f in model_found_files if f in all_files]

        # reflection to correct format
        if len(found_files) == 0:
            corrcted_tpl = format_correct_prompt.format(res=raw_output)
            traj = model.codegen([{"role": "user", "content": corrcted_tpl}], num_samples=1)[0]
            formated_res = ["response"]
            traj['prompt'] = [{"role": "user", "content": corrcted_tpl}]
            file_loc_trajs.append(traj)
            self.logger.info(formated_res)
            model_found_files = self._parse_top5_file(formated_res)
            found_files = [f for f in model_found_files if f in all_files]
            if len(found_files) == 0:
                found_files = [get_best_match(f, all_files) for f in model_found_files]

        # extract the first-order module graph context
        import_content = ""
        _parsed_path = []
        for loc in found_files:
            if loc in _parsed_path:
                continue
            import_content += f"file: {loc}\n {get_imports_of_file(loc, self.instance_id)}\n"
            _parsed_path.append(loc)

        # reflection with module call graph
        reflection_message = [{"role": "user", "content": file_reflection_prompt.format(problem_statement=self.problem_statement,
                                                                       structure=show_project_structure(
                                                                           self.structure).strip(),
                                                                       import_content=import_content,
                                                                       pre_files=found_files)}]
        traj = model.codegen(
            reflection_message,
            num_samples=1)[0]
        traj["prompt"] = reflection_message
        file_loc_trajs.append(traj)
        reflection_result = traj["response"]
        self.logger.info(reflection_result)
        reflection_files = self._parse_top5_file(reflection_result)
        reflection_files = [f for f in reflection_files if f in all_files]
        if len(reflection_files) == 0:
            reflection_files = [get_best_match(f, all_files) for f in reflection_files]

        return (
            reflection_files,
            {"raw_output_files": raw_output},
            traj,
            file_loc_trajs,
        )
    

    def file_localize_with_native_function_call(self, max_retry=10, mock=False,use_cosil=False):
        file_loc_trajs = []
        from repoSearcher.util.api_requests import num_tokens_from_messages
        from repoSearcher.util.model import make_model
        all_files = get_all_of_files(self.instance_id)
        bug_report = bug_report_template.format(problem_statement=self.problem_statement,
                                                structure=show_project_structure(self.structure).strip())

        system_msg = file_system_prompt_without_tool
        guidence_msg = file_guidence_prmpt_without_tool.format(pre_select_num=int(max_retry * 0.75),
                                                               top_n=int(max_retry / 2))
        user_msg = f"""
{bug_report}
{guidence_msg}
{file_summary}
"""

        message = [
            {
                "role": "system",
                "content": system_msg
            },
            {
                "role": "user",
                "content": user_msg
            }
        ]
        self.logger.info(f"prompting with message:\n{message}")
        self.logger.info("=" * 80)
        if mock:
            self.logger.info("Skipping querying model since mock=True")
            traj = {
                "prompt": message,
                "usage": {
                    "prompt_tokens": num_tokens_from_messages(message, self.model_name),
                },
            }
            file_loc_trajs = [traj]
            return [], {"raw_output_loc": ""}, traj, file_loc_trajs

        model = make_model(
            model=self.model_name,
            backend=self.backend,
            logger=self.logger,
            max_tokens=self.max_tokens,
            temperature=0,
            batch_size=1,
        )

        traj = model.codegen(message, num_samples=1)[0]
        traj["prompt"] = message
        raw_output = traj["response"]
        file_loc_trajs.append(traj)

        self.logger.info(raw_output)
        model_found_files = self._parse_top5_file(raw_output)

        import difflib
        def get_best_match(file: str, all_files: list[str], cutoff: float = 0.8) -> str:
            if file in all_files:
                return file
            matches = difflib.get_close_matches(file, all_files, n=1, cutoff=cutoff)
            return matches[0] if matches else file

        found_files = [f for f in model_found_files if f in all_files]

        # reflection to correct format
        if len(found_files) == 0:
            corrcted_tpl = format_correct_prompt.format(res=raw_output)
            traj = model.codegen([{"role": "user", "content": corrcted_tpl}], num_samples=1)[0]
            formated_res = ["response"]
            traj['prompt'] = [{"role": "user", "content": corrcted_tpl}]
            file_loc_trajs.append(traj)
            self.logger.info(formated_res)
            model_found_files = self._parse_top5_file(formated_res)
            found_files = [f for f in model_found_files if f in all_files]
            if len(found_files) == 0:
                found_files = [get_best_match(f, all_files) for f in model_found_files]

        # extract the first-order module graph context
        import_content = ""
        _parsed_path = []
        for loc in found_files:
            if loc in _parsed_path:
                continue
            import_content += f"file: {loc}\n {get_imports_of_file(loc, self.instance_id)}\n"
            _parsed_path.append(loc)

        # reflection with module call graph
        message = [{"role": "user", "content": file_reflection_prompt_native_function_call.format(problem_statement=self.problem_statement,
                                                                       structure=show_project_structure(
                                                                           self.structure).strip(),
                                                                       import_content=import_content,
                                                                       pre_files=found_files) + tool_description}]


        location_summary_tokens = num_tokens_from_messages([{
            "role": "user",
            "content": location_summary_file
        }], self.model_name) + self.max_tokens

        current_tokens = num_tokens_from_messages(message, self.model_name)

        max_try = 10
        finish_flag = False
        # Seach step
        for j in range(max_try):
            if finish_flag:
                break
            # print(current_tokens)
            # breakpoint()
            if current_tokens > self.MAX_CONTEXT_LENGTH - 3 * location_summary_tokens:
                message.pop()
                break
            try:
                traj = model.codegen(message, num_samples=1)[0]
                traj['prompt'] = message.copy()
                file_loc_trajs.append(traj)
                # current_tokens += traj["usage"]["completion_tokens"] + traj["usage"]["prompt_tokens"]
            except Exception as e:
                print(e)
                if "Tokens" in str(e):  # Check if error message indicates context length issue
                    message.pop()
                    break
            content = traj["response"]
            # only get the first action
            if "</tool_call>" in content:
                content = content.split("</tool_call>")[0] + "</tool_call>"
            message.append({
                "role": "assistant",
                "content": content
            })
            current_tokens = num_tokens_from_messages(message, self.model_name)
            
            try:
                from sglang.srt.function_call.function_call_parser import FunctionCallParser
            except ImportError:
                from sglang.srt.function_call_parser import FunctionCallParser
            from sglang.srt.openai_api.protocol import Tool
            from json import JSONDecodeError

            tool_schemas = [{'type': 'function', 'function': {'name': 'get_code_of_class', 'description': 'A tool for geting the class content of specific file', 'parameters': {'type': 'object', 'properties': {'file_name': {'type': 'string', 'description': 'The full file name', 'enum': None}, 'class_name': {'type': 'string', 'description': 'class name', 'enum': None}}, 'required': ['file_name', 'class_name']}, 'strict': True}}, {'type': 'function', 'function': {'name': 'get_code_of_class_function', 'description': 'A tool for geting the function content of specific class', 'parameters': {'type': 'object', 'properties': {'file_name': {'type': 'string', 'description': 'The full file name', 'enum': None}, 'class_name': {'type': 'string', 'description': 'class name', 'enum': None}, 'func_name': {'type': 'string', 'description': 'function name', 'enum': None}}, 'required': ['file_name', 'class_name', 'func_name']}, 'strict': True}}, {'type': 'function', 'function': {'name': 'get_code_of_file_function', 'description': 'A tool for geting the function content of specific file', 'parameters': {'type': 'object', 'properties': {'file_name': {'type': 'string', 'description': 'The full file name', 'enum': None}, 'func_name': {'type': 'string', 'description': 'function name', 'enum': None}}, 'required': ['file_name', 'func_name']}, 'strict': True}}, {'type': 'function', 'function': {'name': 'exit', 'description': 'exit if you have found all the information needed.', 'parameters': {}}}]

            tool_call_parser_type = "qwen25"
            sgl_tools = [Tool.model_validate(tool_schema) for tool_schema in tool_schemas]
            function_call_parser = FunctionCallParser(
                sgl_tools,
                tool_call_parser_type,
            )
            if "exit()" in content:
                finish_flag = True
                break
            if function_call_parser.has_tool_call(content):
                try:
                    normed_content, tool_calls = function_call_parser.parse_non_stream(content)
                except JSONDecodeError:
                    normed_content = content
                    tool_calls = []
                except AttributeError:
                    normed_content = content
                    tool_calls = []
                try:
                    tool_call = tool_calls[0]
                    tool_name = tool_call.name
                    tool_args = json.loads(tool_call.parameters)
                    if tool_name == 'get_code_of_class':
                        tool_args['file_name'] = tool_args['file_name'].replace("'", "").replace('"', "")
                        tool_args['class_name'] = tool_args['class_name'].replace("'", "").replace('"', "")
                        function_call = f"{tool_name}({tool_args['file_name']}, {tool_args['class_name']})"
                        function_retval = get_code_of_class(tool_args['file_name'], tool_args['class_name'], self.instance_id)
                    elif tool_name == 'get_code_of_class_function':
                        tool_args['file_name'] = tool_args['file_name'].replace("'", "").replace('"', "")
                        tool_args['class_name'] = tool_args['class_name'].replace("'", "").replace('"', "")
                        tool_args['func_name'] = tool_args['func_name'].replace("'", "").replace('"', "")
                        function_call = f"{tool_name}({tool_args['file_name']}, {tool_args['class_name']}, {tool_args['func_name']})"
                        function_retval = get_code_of_class_function(tool_args['file_name'], tool_args['class_name'], tool_args['func_name'], self.instance_id)
                    elif tool_name == 'get_code_of_file_function':
                        tool_args['file_name'] = tool_args['file_name'].replace("'", "").replace('"', "")
                        tool_args['func_name'] = tool_args['func_name'].replace("'", "").replace('"', "")
                        function_call = f"{tool_name}({tool_args['file_name']}, {tool_args['func_name']})"
                        function_retval = get_code_of_file_function(tool_args['file_name'], tool_args['func_name'], self.instance_id)
                    elif tool_name == 'exit':
                        finish_flag = True
                        break
                    else:
                        raise ValueError(f"Unknown tool name: {tool_name}")
                
                    check_func_prompt = f"""
Here is a result of a function/class code retrived by '{function_call}'.
<code>
{function_retval}
</code>
Your goal is to locate the culprit locations for the bug. If you need more information, you can call previous functions.
If you are confident of the answer, you can call the ```exit()``` function to finish the information seeking.
"""
                    message.append({
                        "role": "user",
                        "content": check_func_prompt
                    })
                    current_tokens = num_tokens_from_messages(message, self.model_name)
                except Exception as e:
                    print(e)
                    message.append({
                    "role": "user",
                    "content": """wrong tool call format! For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:
For example:
<tool_call>
{"name": "get_code_of_class", "arguments": {    "file_name": "file1.py",    "class_name": "class1"  }}
</tool_call>
Note: all the arguments must be filled in."""})
                current_tokens = num_tokens_from_messages(message, self.model_name)
            else:
                message.append({
                    "role": "user",
                    "content": """wrong tool call format! For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:
For example:
<tool_call>
{"name": "get_code_of_class", "arguments": {    "file_name": "file1.py",    "class_name": "class1"  }}
</tool_call>
Note: all the arguments must be filled in."""})
                current_tokens = num_tokens_from_messages(message, self.model_name)


        while current_tokens > self.MAX_CONTEXT_LENGTH - self.max_tokens - location_summary_tokens and message:
            message.pop()

        # summary the locs
        message.append({
            "role": "user",
            "content": location_summary_file
        })

        traj = model.codegen(
            message,
            num_samples=1)[0]
        traj["prompt"] = message
        file_loc_trajs.append(traj)
        reflection_result = traj["response"]
        self.logger.info(reflection_result)
        reflection_files = self._parse_top5_file(reflection_result)
        reflection_files = [f for f in reflection_files if f in all_files]
        if len(reflection_files) == 0:
            reflection_files = [get_best_match(f, all_files) for f in reflection_files]
        
        '''begin reflection with tools'''

        found_files = [f for f in reflection_files if f in all_files]

        # reflection to correct format
        if len(found_files) == 0:
            corrcted_tpl = format_correct_prompt.format(res=raw_output)
            traj = model.codegen([{"role": "user", "content": corrcted_tpl}], num_samples=1)[0]
            formated_res = ["response"]
            traj['prompt'] = [{"role": "user", "content": corrcted_tpl}]
            file_loc_trajs.append(traj)
            self.logger.info(formated_res)
            model_found_files = self._parse_top5_file(formated_res)
            found_files = [f for f in model_found_files if f in all_files]
            if len(found_files) == 0:
                found_files = [get_best_match(f, all_files) for f in model_found_files]

        # extract the first-order module graph context
        import_content = ""
        _parsed_path = []
        for loc in found_files:
            if loc in _parsed_path:
                continue
            import_content += f"file: {loc}\n {get_imports_of_file(loc, self.instance_id)}\n"
            _parsed_path.append(loc)

        # reflection with module call graph
        reflection_message = [{"role": "user", "content": file_reflection_prompt.format(problem_statement=self.problem_statement,
                                                                       structure=show_project_structure(
                                                                           self.structure).strip(),
                                                                       import_content=import_content,
                                                                       pre_files=found_files)}]
        traj = model.codegen(
            reflection_message,
            num_samples=1)[0]
        traj["prompt"] = reflection_message
        file_loc_trajs.append(traj)
        reflection_result = traj["response"]
        self.logger.info(reflection_result)
        reflection_files = self._parse_top5_file(reflection_result)
        reflection_files = [f for f in reflection_files if f in all_files]
        if len(reflection_files) == 0:
            reflection_files = [get_best_match(f, all_files) for f in reflection_files]

        return (
            reflection_files,
            {"raw_output_files": raw_output},
            traj,
            file_loc_trajs,
        )

    def ablation_file(self, max_retry=10, mock=False):
        from repoSearcher.util.api_requests import num_tokens_from_messages
        from repoSearcher.util.model import make_model
        all_files = get_all_of_files(self.instance_id)
        bug_report = bug_report_template.format(problem_statement=self.problem_statement,
                                                structure=show_project_structure(self.structure).strip())

        system_msg = file_system_prompt_without_tool

        user_msg = f"""
{bug_report}
{file_summary}
"""

        message = [
            {
                "role": "system",
                "content": system_msg
            },
            {
                "role": "user",
                "content": user_msg
            }
        ]
        self.logger.info(f"prompting with message:\n{message}")
        self.logger.info("=" * 80)
        if mock:
            self.logger.info("Skipping querying model since mock=True")
            traj = {
                "prompt": message,
                "usage": {
                    "prompt_tokens": num_tokens_from_messages(message, self.model_name),
                },
            }
            return [], {"raw_output_loc": ""}, traj

        model = make_model(
            model=self.model_name,
            backend=self.backend,
            logger=self.logger,
            max_tokens=self.max_tokens,
            temperature=0,
            batch_size=1,
        )
        try:
            traj = model.codegen(message, num_samples=1)[0]
            traj["prompt"] = message
            raw_output = traj["response"]
        except:
            bug_report = bug_report_template.format(problem_statement=self.problem_statement,
                                                    structure=show_project_structure(self.structure).strip())
            user_msg = f"""
                                 {bug_report}
                                 {file_summary}
                                 """
            message = [{"role": "system", "content": system_msg},
                       {"role": "user", "content": user_msg}]
            traj = model.codegen(message, num_samples=1)[0]
            traj["prompt"] = message
            raw_output = traj["response"]
        model_found_files = self._parse_top5_file(raw_output)

        found_files = [f for f in model_found_files if f in all_files]

        self.logger.info(raw_output)

        return (
            found_files,
            {"raw_output_files": raw_output},
            traj,
        )

    def ablation_func(
            self, max_retry=10, file=None, mock=False,
    ) -> tuple[list[str], Any, Any]:
        # lazy import, not sure if this is actually better?

        from repoSearcher.util.model import make_model
        bug_report = bug_report_template_wo_repo_struct.format(problem_statement=self.problem_statement).strip()
        system_msg = location_system_prompt_ablation
        bug_file_content = self.consturct_bug_file_list(file)
        location_guidence_msg = location_guidence_prmpt_ablation.format(bug_file_list=bug_file_content)
        user_msg = f"""
                {bug_report}
                {location_guidence_msg}
                {location_summary_ablation}
                """

        message = [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg}
        ]

        model = make_model(
            model=self.model_name,
            backend=self.backend,
            logger=self.logger,
            max_tokens=self.max_tokens,
            temperature=0.85,
            batch_size=1,
        )
        traj = model.codegen(message, num_samples=1)[0]
        traj["prompt"] = message
        raw_output = traj["response"]

        self.logger.info(raw_output)
        model_found_locs = extract_code_blocks(raw_output)
        model_found_locs_separated = extract_locs_for_files(
            model_found_locs, file
        )

        return (
            model_found_locs_separated,
            raw_output,
            traj,
        )

    def ablation_refection(self, max_retry=10, mock=False):
        from repoSearcher.util.api_requests import num_tokens_from_messages
        from repoSearcher.util.model import make_model
        all_files = get_all_of_files(self.instance_id)
        bug_report = bug_report_template.format(problem_statement=self.problem_statement,
                                                structure=show_project_structure(self.structure).strip())

        system_msg = file_system_prompt_without_tool

        user_msg = f"""
        {bug_report}
        {file_summary}
        """

        message = [
            {
                "role": "system",
                "content": system_msg
            },
            {
                "role": "user",
                "content": user_msg
            }
        ]
        self.logger.info(f"prompting with message:\n{message}")
        self.logger.info("=" * 80)
        if mock:
            self.logger.info("Skipping querying model since mock=True")
            traj = {
                "prompt": message,
                "usage": {
                    "prompt_tokens": num_tokens_from_messages(message, self.model_name),
                },
            }
            return [], {"raw_output_loc": ""}, traj

        model = make_model(
            model=self.model_name,
            backend=self.backend,
            logger=self.logger,
            max_tokens=self.max_tokens,
            temperature=0,
            batch_size=1,
        )

        traj = model.codegen(message, num_samples=1)[0]
        traj["prompt"] = message
        raw_output = traj["response"]

        self.logger.info(raw_output)
        model_found_files = self._parse_top5_file(raw_output)

        # extract the first-order module graph context
        import_content = ""
        _parsed_path = []
        for loc in model_found_files:
            if loc in _parsed_path:
                continue
            import_content += f"file: {loc}\n {get_imports_of_file(loc, self.instance_id)}\n"
            _parsed_path.append(loc)

        # reflection with model call graph
        reflection_result = model.codegen(
            [{"role": "user", "content": file_reflection_prompt.format(problem_statement=self.problem_statement,
                                                                       structure=show_project_structure(
                                                                           self.structure).strip(),
                                                                       import_content=import_content,
                                                                       pre_files=model_found_files)}],
            num_samples=1)[0]["response"]
        self.logger.info(reflection_result)
        reflection_files = self._parse_top5_file(reflection_result)


        return (
            reflection_files,
            {"raw_output_files": raw_output},
            traj,
        )

def construct_topn_file_context(
        file_to_locs,
        pred_files,
        file_contents,
        structure,
        context_window: int,
        loc_interval: bool = True,
        fine_grain_loc_only: bool = False,
        add_space: bool = False,
        sticky_scroll: bool = False,
        no_line_number: bool = True,
):
    """Concatenate provided locations to form a context.

    loc: {"file_name_1": ["loc_str_1"], ...}
    """
    file_loc_intervals = dict()
    topn_content = ""

    for pred_file, locs in file_to_locs.items():
        content = file_contents[pred_file]
        line_locs, context_intervals = transfer_arb_locs_to_locs(
            locs,
            structure,
            pred_file,
            context_window,
            loc_interval,
            fine_grain_loc_only,
            file_content=file_contents[pred_file] if pred_file in file_contents else "",
        )

        if len(line_locs) > 0:
            # Note that if no location is predicted, we exclude this file.
            file_loc_content = line_wrap_content(
                content,
                context_intervals,
                add_space=add_space,
                no_line_number=no_line_number,
                sticky_scroll=sticky_scroll,
            )
            topn_content += f"### {pred_file}\n{file_loc_content}\n\n\n"
            file_loc_intervals[pred_file] = context_intervals

    return topn_content, file_loc_intervals

# if __name__ == '__main__':
#
#     from datasets import load_from_disk
#     print("加载数据")
#     swe_bench_data = load_from_disk("../../datasets/SWE-bench_Lite_test")
#     # bug = swe_bench_data[5]
#     bug = [x for x in swe_bench_data if x["instance_id"] == "django__django-13315"][0]
#     problem_statement = bug["problem_statement"]
#     instance_id = bug["instance_id"]
#     print(problem_statement)
#     print(instance_id)
#     d = load_json(f"../../repo_structures/{instance_id}.json")
#     structure = d["structure"]
#     # print(show_project_structure(structure))
#     found_files = ["django/forms/models.py"]
#     import_content = ""
#     _parsed_path = []
#     for loc in found_files:
#         if loc in _parsed_path:
#             continue
#         import_content += f"file: {loc}\n {get_imports_of_file(loc, instance_id)}\n"
#         _parsed_path.append(loc)
#
#     print(import_content)
#     """
#     file: django/db/models/fields/related.py
#     imports: ['django.forms', 'django.apps', 'django.conf', 'django.core', 'django.db.model', 'django.db.backends'...]
#     """
#     def consturct_bug_file_list(file: list):
#         bug_file_content = ""
#         for name in file:
#             class_content = get_classes_of_file(name, instance_id)
#             class_list = eval(get_classes_of_file(name, instance_id))
#             class_func_content = "[\n"
#             for class_name in class_list:
#                 class_func = get_functions_of_class(class_name, instance_id)
#                 class_func_content += f"{class_name}: {class_func} \n"
#             class_func_content += "]"
#             file_func_content = get_functions_of_file(name, instance_id)
#             single_file_context = f"file: {name} \n\t class: {class_content} \n\t static functions:  {file_func_content} \n\t class fucntions: {class_func_content}\n"
#             bug_file_content += single_file_context
#             # print(bug_file_content)
#         return bug_file_content
#     bug_file_content = consturct_bug_file_list(found_files)
#     print(bug_file_content)


    # import logging
    # fl = AFL(
    #     d["instance_id"],
    #     structure,
    #     problem_statement,
    #     "deepseek-coder",
    #     "deepseek",
    #     logging.getLogger("AFL"),
    # )
    # print("------------------------------")
    # print("start localization")
    # ret = fl.file_localize()[0]
    # print(ret)
    # loc = fl.localize(file=ret)[0]
    # print(loc)
    # line_loc = fl.localize_line(ret, loc, temperature=0.85, num_samples=1)[0]
    # print(line_loc)