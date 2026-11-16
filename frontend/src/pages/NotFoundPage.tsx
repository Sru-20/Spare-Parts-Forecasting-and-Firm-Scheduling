import { Link } from "react-router-dom";
import { PageHeader, StateCard } from "../design-system/ui";

export function NotFoundPage() {
  return (
    <>
      <PageHeader eyebrow="NOVA WORKSPACE" title="Page not found" description="This route is not part of the planning workspace." />
      <StateCard
        title="The page does not exist"
        description="Use the workspace navigation to return to a saved planning view."
        action={<Link className="button button-primary" to="/">Go to overview</Link>}
      />
    </>
  );
}
